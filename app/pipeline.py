"""Run test-bed analysis for one or more nights (backend orchestration)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from app.run_status import RunStatus, model_step_label
from backend.conditioning import ConditioningParams, condition_night
from backend.config import DataSource
from backend.data_loader import load_night_data
from backend.engines import engine_for_night, engine_s3_key, load_ptl, load_ptl_by_key
from backend.evaluator import Evaluator, StimulationParams
from backend.expert_annotations import expert_events_to_records, load_expert_annotations
from backend.metrics import calculate_audio_metrics, calculate_spo2_metrics, classification_metrics
from backend.features import (
    DEFAULT_INPUT_FEATURES,
    available_features,
    feature_set_key,
    features_require_retrain,
    normalize_feature_list,
    validate_requested_features,
)
from backend.ml import ModelRunConfig, evaluate_windows, train_model
from backend.scoring import extract_flatline_times
from backend.sliding_inference import run_model_detections, scoring_detections_from_onsets
from backend.temp_cleanup import cleanup_run_artifacts
from backend.training_windows import (
    TrainingWindow,
    generate_windows_from_night,
    split_windows,
)
from backend.types import NightRef
from backend.config import DEFAULT_SAMPLE_RATE
from backend.window_config import (
    assert_s3_ptl_window_compatible,
    scaled_onset_margins_sec,
    scoring_onset_margins,
    window_generation_params_from_scoring,
    window_samples_for_scoring,
)
from parameters import ParameterSet


def ui_config_to_parameter_set(ui: Dict[str, Any]) -> ParameterSet:
    """Build ParameterSet from Streamlit widget values."""
    model_mode_map = {
        "Scoring only (no model)": "none",
        "Use existing S3 engine (.ptl)": "s3_ptl",
        "Train new model": "train",
    }
    arch_map = {"CNN": "cnn", "LSTM": "lstm", "GLM": "glm", "XGBoost": "xgboost"}
    window_sec = int(ui["window_sec"])
    sec_before, sec_after = scaled_onset_margins_sec(window_sec)

    return ParameterSet(config_dict={
        "conditioning": {
            "spo2_interpolation_window": int(ui["spo2_window_sec"]),
            "audio_conditioning_steps": list(ui.get("audio_conditioning_steps", [])),
            "slope_threshold": float(ui.get("slope_threshold", 0.025)),
            "scale_factor_high": float(ui.get("scale_factor_high", 1.0)),
            "scale_factor_low": float(ui.get("scale_factor_low", 0.75)),
        },
        "scoring": {
            "variance_threshold": float(ui["variance_threshold"]),
            "min_apnea_seconds": int(ui.get("min_apnea_seconds", 10)),
            "sliding_window_width": window_sec,
            "sec_before_onset": sec_before,
            "sec_after_onset": sec_after,
        },
        "training": {
            "test_frac": float(ui.get("test_frac", 0.2)),
            "split_strategy": ui.get("split_strategy", "random"),
            "balance_classes": True,
            "input_features": list(
                normalize_feature_list(ui.get("input_features", DEFAULT_INPUT_FEATURES))
            ),
        },
        "model": {
            "mode": model_mode_map.get(ui.get("model_mode"), ui.get("model_mode", "none")),
            "architecture": arch_map.get(ui.get("model_architecture", "CNN"), "cnn"),
            "engine_s3_key": ui.get("engine_s3_key"),
            "auto_match_engine": bool(ui.get("auto_match_engine", True)),
            "epochs": int(ui.get("train_epochs", 5)),
            "batch_size": int(ui.get("train_batch_size", 32)),
            "learning_rate": float(ui.get("train_learning_rate", 0.001)),
            "early_stopping_patience": int(ui.get("early_stopping_patience", 5)),
            "glm_C": float(ui.get("glm_C", 1.0)),
            "glm_max_iter": int(ui.get("glm_max_iter", 1000)),
            "glm_penalty": ui.get("glm_penalty", "l2"),
            "glm_solver": ui.get("glm_solver", "lbfgs"),
            "xgb_n_estimators": int(ui.get("xgb_n_estimators", 100)),
            "xgb_max_depth": int(ui.get("xgb_max_depth", 6)),
            "xgb_learning_rate": float(ui.get("xgb_learning_rate", 0.1)),
            "xgb_subsample": float(ui.get("xgb_subsample", 0.8)),
            "xgb_colsample_bytree": float(ui.get("xgb_colsample_bytree", 0.8)),
            "xgb_min_child_weight": float(ui.get("xgb_min_child_weight", 1.0)),
        },
        "prediction": {
            "detection_algorithm": "live_postprocessing",
            "activation_threshold": float(ui.get("activation_threshold", 0.7)),
            "activation_seconds": int(ui.get("activation_sec", 3)),
            "seconds_since_last_treatment": float(ui.get("cooldown_sec", 5.0)),
            "inference_interval_sec": float(ui.get("inference_interval_sec", 1.0)),
            "haptic_duration": int(ui.get("haptic_ms", 1000)),
            "actuation_start_delay_min": float(ui.get("actuation_start_delay_min", 0)),
        },
    })


def model_config_from_parameter_set(param_set: ParameterSet) -> ModelRunConfig:
    import torch

    cfg = param_set.config
    m = cfg.get("model", {})
    train = cfg.get("training", {})
    scoring = cfg.get("scoring", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_features = normalize_feature_list(train.get("input_features", DEFAULT_INPUT_FEATURES))
    return ModelRunConfig(
        mode=m.get("mode", "none"),
        model_type=m.get("architecture", "cnn"),
        engine_s3_key=m.get("engine_s3_key"),
        auto_match_engine=bool(m.get("auto_match_engine", True)),
        epochs=int(m.get("epochs", 5)),
        batch_size=int(m.get("batch_size", 32)),
        learning_rate=float(m.get("learning_rate", 0.001)),
        test_frac=float(train.get("test_frac", 0.2)),
        early_stopping_patience=int(m.get("early_stopping_patience", 5)),
        device=device,
        input_features=input_features,
        window_samples=window_samples_for_scoring(scoring, DEFAULT_SAMPLE_RATE),
        glm_C=float(m.get("glm_C", 1.0)),
        glm_max_iter=int(m.get("glm_max_iter", 1000)),
        glm_penalty=str(m.get("glm_penalty", "l2")),
        glm_solver=str(m.get("glm_solver", "lbfgs")),
        xgb_n_estimators=int(m.get("xgb_n_estimators", 100)),
        xgb_max_depth=int(m.get("xgb_max_depth", 6)),
        xgb_learning_rate=float(m.get("xgb_learning_rate", 0.1)),
        xgb_subsample=float(m.get("xgb_subsample", 0.8)),
        xgb_colsample_bytree=float(m.get("xgb_colsample_bytree", 0.8)),
        xgb_min_child_weight=float(m.get("xgb_min_child_weight", 1.0)),
    )


def _model_config_for_prep(
    model_config: ModelRunConfig,
    prep: Dict[str, Any],
) -> ModelRunConfig:
    """Apply per-night sample rate and window length from prepare_night."""
    cfg = replace(
        model_config,
        window_samples=int(prep["window_samples"]),
    )
    if cfg.mode == "s3_ptl":
        assert_s3_ptl_window_compatible(cfg.window_samples)
    return cfg


def resolve_model_config(
    model_config: ModelRunConfig,
    param_set: ParameterSet,
) -> Tuple[ModelRunConfig, Dict[str, Any]]:
    """
    Apply feature-set policy: non-default features force training (S3 engines are audio-only).
    """
    train = param_set.config.get("training", {})
    input_features = normalize_feature_list(train.get("input_features", model_config.input_features))
    meta: Dict[str, Any] = {
        "input_features": list(input_features),
        "feature_set_key": feature_set_key(input_features),
    }
    cfg = replace(model_config, input_features=input_features)
    if features_require_retrain(input_features) and cfg.mode == "s3_ptl":
        meta["forced_train"] = True
        meta["forced_train_reason"] = "non_default_input_features"
        cfg = replace(cfg, mode="train")
    return cfg, meta


def prepare_night(
    night_ref: NightRef,
    param_set: ParameterSet,
    subset: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Load, condition, score, and build train/val windows (no model step)."""
    config = param_set.config
    scoring = config["scoring"]
    training = config["training"]

    night = load_night_data(night_ref)
    cond_params = ConditioningParams.for_source(night_ref.source, config)
    cond_params.is_csv_audio = night_ref.primary_key.lower().endswith(".csv")
    conditioned = condition_night(night, params=cond_params, config=config)

    df = conditioned.df
    if subset is not None:
        start_m, end_m = subset
        t0, t1 = start_m * 60.0, end_m * 60.0
        df = df[(df["time"] >= t0) & (df["time"] <= t1)].copy()
        if df.empty:
            raise ValueError(f"No data in subset {start_m}-{end_m} minutes")

    sec_before, sec_after = scoring_onset_margins(scoring)
    win_params = window_generation_params_from_scoring(
        scoring, conditioned.sample_rate, night_ref.source
    )
    flatline_times, onset_times = extract_flatline_times(
        df,
        sample_rate=conditioned.sample_rate,
        variance_threshold=float(scoring["variance_threshold"]),
        min_apnea_seconds=int(scoring.get("min_apnea_seconds", 10)),
        seconds_before_apnea=sec_before,
        seconds_after_apnea=sec_after,
    )

    night.audio = df
    night.sample_rate = conditioned.sample_rate
    night.total_seconds = float(df["time"].iloc[-1]) if len(df) else 0.0

    input_features = validate_requested_features(
        training.get("input_features", DEFAULT_INPUT_FEATURES),
        df,
        night_ref.source,
        night_id=night_ref.night_id,
    )

    windows = generate_windows_from_night(
        night,
        flatline_times,
        params=win_params,
        input_features=input_features,
    )

    split_strategy = training.get("split_strategy", "random")
    test_frac = float(training.get("test_frac", 0.2))
    night_ids = [night_ref.night_id] * len(windows)
    train_w, val_w = split_windows(
        windows,
        strategy=split_strategy,
        test_frac=test_frac,
        night_ids=night_ids if split_strategy != "random" else None,
    )

    return {
        "night_ref": night_ref,
        "night": night,
        "conditioned_df": df,
        "sample_rate": conditioned.sample_rate,
        "window_samples": win_params.window_samples,
        "window_sec": int(round(win_params.flatline_window_sec)),
        "sec_before_onset": win_params.sec_before,
        "sec_after_onset": win_params.sec_after,
        "flatline_times": flatline_times,
        "onset_times": onset_times,
        "windows": windows,
        "train_windows": train_w,
        "val_windows": val_w,
        "input_features": list(input_features),
        "available_features": list(available_features(df, night_ref.source)),
    }


def _baseline_val_metrics(train_w: List[TrainingWindow], val_w: List[TrainingWindow]) -> Dict[str, Any]:
    val_metrics: Dict[str, Any] = {"n_val_windows": len(val_w)}
    if val_w:
        val_labels = [w.label for w in val_w]
        val_metrics["label_counts"] = {str(k): val_labels.count(k) for k in sorted(set(val_labels))}
    if train_w and val_w:
        majority = max(set(w.label for w in train_w), key=lambda l: sum(1 for w in train_w if w.label == l))
        preds = [majority] * len(val_w)
        true = [w.label for w in val_w]
        val_metrics.update(
            classification_metrics(preds, true, class_names=["non-apnea", "onset-apnea", "flatline"])
        )
        val_metrics["mode"] = "baseline_majority"
    return val_metrics


def _run_model_on_windows(
    night_ref: NightRef,
    train_w: List[TrainingWindow],
    val_w: List[TrainingWindow],
    model_config: ModelRunConfig,
    trained_model: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Any], bool, str]:
    """
    Returns (val_metrics, model_info, model, is_jit, model_type_for_inference).

    model is set when mode is s3_ptl or train (for sliding inference in finalize).
    """
    info: Dict[str, Any] = {
        "mode": model_config.mode,
        "input_features": list(model_config.input_features),
        "feature_set_key": model_config.feature_set_key,
        "n_channels": model_config.n_channels,
    }
    is_jit = False
    mtype: str = model_config.model_type
    n_ch = model_config.n_channels

    if model_config.mode == "none":
        return _baseline_val_metrics(train_w, val_w), info, None, False, mtype

    if model_config.mode == "train":
        if trained_model is not None:
            model = trained_model
            info["mode"] = "train_pooled"
        else:
            model, metrics = train_model(train_w, val_w, model_config)
            info["training_history"] = metrics.get("training_history", [])
            info["model_type"] = model_config.model_type
            info["inference_model"] = "trained"
            return metrics, info, model, False, model_config.model_type

        metrics = evaluate_windows(
            model,
            val_w,
            mtype,
            device=model_config.device,
            is_jit=False,
            n_channels=n_ch,
            window_samples=model_config.window_samples,
        )
        info["model_type"] = mtype
        return metrics, info, model, False, mtype

    if model_config.mode == "s3_ptl":
        mtype = "cnn"  # production .ptl exports are mobile CNN
        is_jit = True
        if model_config.engine_s3_key:
            model, local_path = load_ptl_by_key(
                night_ref.bucket,
                model_config.engine_s3_key,
                device=model_config.device,
            )
            info["engine_s3_key"] = model_config.engine_s3_key
        else:
            eng = engine_for_night(night_ref)
            if eng is None:
                expected = engine_s3_key(night_ref)
                raise FileNotFoundError(
                    f"No engine at s3://{night_ref.bucket}/{expected}. "
                    "Pick another engine or train a new model."
                )
            model, local_path = load_ptl(eng, device=model_config.device)
            info["engine_s3_key"] = eng.s3_key
            info["engine_night_id"] = eng.night_id

        info["local_ptl_path"] = local_path
        info["model_type"] = mtype
        metrics = evaluate_windows(
            model,
            val_w,
            mtype,
            device=model_config.device,
            is_jit=True,
            n_channels=n_ch,
            window_samples=model_config.window_samples,
        )
        return metrics, info, model, is_jit, mtype

    return _baseline_val_metrics(train_w, val_w), info, None, False, mtype


def _build_detections(
    prep: Dict[str, Any],
    param_set: ParameterSet,
    model_config: ModelRunConfig,
    model: Optional[Any],
    is_jit: bool,
    model_type: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Scoring-only uses onset algorithm; model modes use full-night sliding inference."""
    config = param_set.config
    onset_times = prep["onset_times"]

    if model_config.mode == "none" or model is None:
        dets = scoring_detections_from_onsets(onset_times)
        return dets, {"detection_source": "scoring", "n_detections": len(dets)}

    detections, meta = run_model_detections(
        prep["conditioned_df"],
        model,
        model_config,
        is_jit=is_jit,
        sample_rate=int(prep["sample_rate"]),
        prediction_config=config.get("prediction", {}),
    )
    meta["inference_model_type"] = model_type
    return detections, meta


def finalize_night_result(
    prep: Dict[str, Any],
    param_set: ParameterSet,
    model_config: Optional[ModelRunConfig] = None,
    model: Optional[Any] = None,
    is_jit: bool = False,
    model_type: str = "cnn",
) -> Dict[str, Any]:
    """Expert comparison, efficacy, and result dict assembly."""
    night_ref: NightRef = prep["night_ref"]
    df = prep["conditioned_df"]
    val_metrics = prep.get("val_metrics", {})
    model_info = prep.get("model_info", {})

    if model_config is None:
        model_config = ModelRunConfig(mode="none")

    detections, detection_meta = _build_detections(
        prep, param_set, model_config, model, is_jit, model_type
    )
    model_info = {**model_info, **detection_meta}

    expert_eval = None
    if night_ref.source in (DataSource.SCIDB, DataSource.MESA):
        expert_df = load_expert_annotations(night_ref)
        if expert_df is not None and not expert_df.empty:
            events = expert_events_to_records(expert_df)
            expert_eval = Evaluator.eval_expert_annotations(detections, events)
            expert_eval["detection_source"] = detection_meta.get("detection_source", "unknown")

    config = param_set.config
    pred_cfg = config.get("prediction", {})
    det_source = detection_meta.get("detection_source", "scoring")
    at_actuation = det_source == "live_postprocessing"
    stim_params = StimulationParams.from_prediction_config(
        pred_cfg,
        detection_at_actuation_time=at_actuation,
    )
    efficacy = Evaluator.eval_simulated_efficacy(detections, df, stimulation=stim_params)
    efficacy["detection_source"] = det_source

    return {
        "night_id": night_ref.night_id,
        "source": night_ref.source.value,
        "signal": night_ref.signal,
        "subject": night_ref.subject,
        "session": night_ref.session,
        "sample_rate": prep["sample_rate"],
        "total_seconds": prep["night"].total_seconds,
        "n_flatlines": len(prep["flatline_times"]),
        "n_onsets": len(prep["onset_times"]),
        "n_windows": len(prep["windows"]),
        "n_train": len(prep["train_windows"]),
        "n_val": len(prep["val_windows"]),
        "n_detections": len(detections),
        "flatline_times": prep["flatline_times"],
        "onset_times": prep["onset_times"],
        "conditioned_df": df,
        "audio_metrics": calculate_audio_metrics(df),
        "spo2_metrics": calculate_spo2_metrics(df),
        "val_metrics": val_metrics,
        "model_info": model_info,
        "expert_eval": expert_eval,
        "simulated_efficacy": efficacy,
        "has_expert": expert_eval is not None,
        "detection_source": detection_meta.get("detection_source"),
    }


def analyze_night(
    night_ref: NightRef,
    param_set: ParameterSet,
    subset: Optional[Tuple[float, float]] = None,
    model_config: Optional[ModelRunConfig] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Full path for one night including optional model train/inference."""
    if model_config is None:
        model_config = model_config_from_parameter_set(param_set)
    model_config, feature_meta = resolve_model_config(model_config, param_set)

    status = RunStatus(status_callback)
    try:
        status.checkpoint("Load & condition", night_ref.night_id)
        prep = prepare_night(night_ref, param_set, subset=subset)
        prep["model_info"] = {**feature_meta, **prep.get("model_info", {})}
        model_config = _model_config_for_prep(model_config, prep)

        status.checkpoint(model_step_label(model_config.mode), night_ref.night_id)
        val_metrics, model_info, model, is_jit, mtype = _run_model_on_windows(
            night_ref,
            prep["train_windows"],
            prep["val_windows"],
            model_config,
        )
        prep["val_metrics"] = val_metrics
        prep["model_info"] = {**prep.get("model_info", {}), **model_info}

        status.checkpoint("Post-processing", night_ref.night_id)
        return finalize_night_result(
            prep, param_set, model_config=model_config, model=model, is_jit=is_jit, model_type=mtype
        )
    finally:
        cleanup_run_artifacts()


def analyze_batch(
    night_refs: List[NightRef],
    param_set: ParameterSet,
    progress_callback=None,
    model_config: Optional[ModelRunConfig] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run analysis across nights; pools training windows when training on multiple nights."""
    if model_config is None:
        model_config = model_config_from_parameter_set(param_set)
    model_config, feature_meta = resolve_model_config(model_config, param_set)

    status = RunStatus(status_callback)
    per_night: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    preps: List[Dict[str, Any]] = []

    try:
        n_nights = len(night_refs)
        status.checkpoint("Load & condition", f"{n_nights} night(s)")
        for i, ref in enumerate(night_refs):
            if progress_callback:
                progress_callback(i, n_nights, ref.night_id, "preparing")
            try:
                preps.append(prepare_night(ref, param_set))
            except Exception as exc:
                errors.append({"night_id": ref.night_id, "error": str(exc)})

        if not preps:
            return {
                "per_night": [],
                "errors": errors,
                "n_success": 0,
                "n_failed": len(errors),
                "config": param_set.to_dict(),
            }

        trained_model = None
        pooled_info: Dict[str, Any] = dict(feature_meta)
        pooled_mtype = model_config.model_type

        if model_config.mode == "train" and len(preps) > 1:
            all_windows: List[TrainingWindow] = []
            night_ids: List[str] = []
            for p in preps:
                all_windows.extend(p["windows"])
                night_ids.extend([p["night_ref"].night_id] * len(p["windows"]))

            train_cfg = _model_config_for_prep(model_config, preps[0])
            train_w, val_w = split_windows(
                all_windows,
                strategy=param_set.config["training"].get("split_strategy", "random"),
                test_frac=train_cfg.test_frac,
                night_ids=night_ids
                if param_set.config["training"].get("split_strategy") != "random"
                else None,
            )
            status.checkpoint("Training (pooled)", f"{len(train_w)} train / {len(val_w)} val windows")
            try:
                trained_model, pooled_metrics = train_model(train_w, val_w, train_cfg)
                pooled_mtype = train_cfg.model_type
                pooled_info = {
                    "pooled_train": True,
                    "n_train_windows": len(train_w),
                    "n_val_windows": len(val_w),
                    "pooled_val_metrics": pooled_metrics,
                }
            except Exception as exc:
                errors.append({"night_id": "pooled_train", "error": str(exc)})

        model_label = model_step_label(model_config.mode)
        for i, prep in enumerate(preps):
            ref = prep["night_ref"]
            if progress_callback:
                progress_callback(i, len(preps), ref.night_id, "evaluating")
            try:
                night_model_cfg = _model_config_for_prep(model_config, prep)
                if trained_model is not None:
                    status.checkpoint(model_label, f"{ref.night_id} ({i + 1}/{len(preps)})")
                    val_metrics, model_info, model, is_jit, mtype = _run_model_on_windows(
                        ref,
                        prep["train_windows"],
                        prep["val_windows"],
                        night_model_cfg,
                        trained_model=trained_model,
                    )
                    model_info.update(pooled_info)
                else:
                    status.checkpoint(model_label, f"{ref.night_id} ({i + 1}/{len(preps)})")
                    val_metrics, model_info, model, is_jit, mtype = _run_model_on_windows(
                        ref,
                        prep["train_windows"],
                        prep["val_windows"],
                        night_model_cfg,
                    )
                prep["val_metrics"] = val_metrics
                prep["model_info"] = model_info
                status.checkpoint("Post-processing", f"{ref.night_id} ({i + 1}/{len(preps)})")
                per_night.append(
                    finalize_night_result(
                        prep,
                        param_set,
                        model_config=night_model_cfg,
                        model=model,
                        is_jit=is_jit,
                        model_type=mtype,
                    )
                )
            except Exception as exc:
                errors.append({"night_id": ref.night_id, "error": str(exc)})

        return {
            "per_night": per_night,
            "errors": errors,
            "n_success": len(per_night),
            "n_failed": len(errors),
            "config": param_set.to_dict(),
            "pooled_training": pooled_info,
            "feature_meta": feature_meta,
        }
    finally:
        cleanup_run_artifacts()
