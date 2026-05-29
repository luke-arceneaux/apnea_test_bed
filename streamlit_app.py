"""
Sleep Data Parameter Testbed
Streamlit UI wired to backend: S3 nights, conditioning, scoring, windows, evaluation.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import plotly.express as px
import streamlit as st

from app.charts import plot_confusion_matrix, plot_label_distribution, plot_signal_preview
from backend.conditioning import AUDIO_CONDITIONING_STEP_ORDER, audio_steps_from_config
from app.pipeline import (
    analyze_batch,
    analyze_night,
    model_config_from_parameter_set,
    resolve_model_config,
    ui_config_to_parameter_set,
)
from backend.features import DEFAULT_INPUT_FEATURES, features_require_retrain
from backend.config import DEFAULT_SAMPLE_RATE, DataSource
from backend.data_loader import list_available_nights
from backend.engines import engine_for_night, list_s3_engines
from backend.s3_client import _credentials
from backend.metadata_summary import summaries_for_results
from backend.session_metadata import (
    NightRefWithMeta,
    _DEFAULT_MIN_DURATION_MIN,
    catalog_to_display_dataframe,
    default_date_range,
    list_nights_with_metadata,
    load_session_metadata,
    metadata_for_source,
)
from backend.types import NightRef
from parameters import ParameterSet

WINDOW_CLASS_LEGEND = "**0** non-apnea · **1** onset apnea · **2** flatline"
AUDIO_STEP_LABELS = {
    "savgol": "Savitzky-Golay",
    "normalization": "Normalization (min-max)",
    "standardization": "Standardization (z-score)",
    "nonlinear": "Nonlinear scaling",
}
SIGNAL_OVERLAY_LEGEND = (
    "Shaded regions: **onset** (green) — window around apnea start; "
    "**flatline** (orange) — low-variance audio segment."
)
DETECTION_SOURCE_LABELS = {
    "scoring": "Scoring (onsets)",
    "live_postprocessing": "Model + live rules",
}
MODEL_ARCHITECTURES = ["CNN", "LSTM", "GLM", "XGBoost"]
MODEL_ARCH_TO_BACKEND = {
    "CNN": "cnn",
    "LSTM": "lstm",
    "GLM": "glm",
    "XGBoost": "xgboost",
}
MODEL_ARCH_FROM_BACKEND = {v: k for k, v in MODEL_ARCH_TO_BACKEND.items()}

# ---------------------------------------------------------------------------
# Page config & state
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sleep Parameter Testbed",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stMetric { background-color: #f0f2f6; padding: 8px; border-radius: 6px; }
    .panel { background-color: #e8eaf6; padding: 12px; border-radius: 8px;
             border-left: 4px solid #3f51b5; margin: 8px 0; }
    .ok { background-color: #e8f5e9; border-left-color: #4caf50; }
    .warn { background-color: #fff3e0; border-left-color: #ff9800; }
</style>
""",
    unsafe_allow_html=True,
)

for key, default in (
    ("test_results", None),
    ("nights_cache", {}),
    ("metadata_cache", {}),
    ("engines_cache", {}),
    ("selected_night_ids", []),
    ("run_requested", False),
):
    if key not in st.session_state:
        st.session_state[key] = default


def aws_configured() -> bool:
    try:
        _credentials()
        return True
    except EnvironmentError:
        return False


def source_from_label(label: str) -> DataSource:
    return DataSource(label.lower())


def night_label(ref: NightRef) -> str:
    return f"{ref.dataset} | subj{ref.subject} | sess{ref.session} ({ref.signal})"


def refs_from_ids(all_refs: List[NightRef], ids: List[str]) -> List[NightRef]:
    by_id = {r.night_id: r for r in all_refs}
    return [by_id[i] for i in ids if i in by_id]


def fetch_nights(source: DataSource, signal: Optional[str]) -> List[NightRef]:
    cache_key = f"{source.value}:{signal or 'all'}"
    if cache_key in st.session_state.nights_cache:
        return st.session_state.nights_cache[cache_key]
    refs = list_available_nights(source, signal=signal or None)
    st.session_state.nights_cache[cache_key] = refs
    return refs


def fetch_metadata(source: DataSource) -> pd.DataFrame:
    cache_key = source.value
    if cache_key in st.session_state.metadata_cache:
        return st.session_state.metadata_cache[cache_key]
    df = load_session_metadata(source)
    st.session_state.metadata_cache[cache_key] = df
    return df


def parse_date_range(
    date_range: Union[date, Tuple[date, ...], List[date]],
    fallback_start: date,
    fallback_end: date,
) -> Tuple[date, date]:
    """Normalize Streamlit date_input return value (dashboard parity)."""
    if isinstance(date_range, (tuple, list)):
        if len(date_range) == 2:
            return date_range[0], date_range[1]
        if len(date_range) == 1:
            return date_range[0], date_range[0]
        return fallback_start, fallback_end
    return date_range, date_range


def fetch_engines(source: DataSource, dataset: str, signal: str):
    cache_key = f"{source.value}:{dataset}:{signal}"
    if cache_key in st.session_state.engines_cache:
        return st.session_state.engines_cache[cache_key]
    engines = list_s3_engines(source, dataset=dataset, signal=signal)
    st.session_state.engines_cache[cache_key] = engines
    return engines


def parse_subset(segment_str: str) -> Optional[tuple[float, float]]:
    if not segment_str or segment_str.strip().lower() == "full":
        return None
    part = segment_str.strip().split("-")
    if len(part) != 2:
        return None
    return float(part[0]), float(part[1])


# ---------------------------------------------------------------------------
# Sidebar: connection & data catalog
# ---------------------------------------------------------------------------
catalog_entries: List[NightRefWithMeta] = []
all_refs: List[NightRef] = []

with st.sidebar:
    st.header("Connection")
    if aws_configured():
        st.success("AWS credentials detected")
    else:
        st.error("Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        st.caption("Restart Streamlit after exporting env vars.")

    st.divider()
    st.subheader("Data catalog")
    source_label = st.selectbox("Data source", ["Zephyr", "SCIDB", "MESA"], index=0)
    source = source_from_label(source_label)

    signals = {
        DataSource.ZEPHYR: ["stereo"],
        DataSource.SCIDB: ["cannula"],
        DataSource.MESA: ["Flow"],
    }[source]
    signal = st.selectbox("Signal", signals)

    s3_ref_count = 0
    if aws_configured():
        try:
            s3_refs = fetch_nights(source, signal)
            s3_ref_count = len(s3_refs)
            meta_df = fetch_metadata(source)
            scoped_meta = metadata_for_source(meta_df, source)

            if scoped_meta.empty:
                st.warning("No session metadata for this source.")
                if st.button("Refresh night list", key="refresh_night_list", use_container_width=True):
                    st.session_state.nights_cache = {}
                    st.session_state.metadata_cache = {}
                    st.session_state.engines_cache = {}
            else:
                min_date_in_data = scoped_meta["Date"].min()
                max_date_in_data = scoped_meta["Date"].max()
                default_start, default_end = default_date_range(scoped_meta)
                max_duration = int(scoped_meta["Duration (min)"].max())

                date_range = st.date_input(
                    "Session date range",
                    value=(default_start, default_end),
                    min_value=min_date_in_data,
                    max_value=max_date_in_data,
                    key=f"session_date_range_{source.value}",
                )
                start_date, end_date = parse_date_range(
                    date_range, default_start, default_end
                )
                min_duration = st.number_input(
                    "Minimum duration (minutes)",
                    min_value=0,
                    max_value=max(max_duration, 1),
                    value=int(_DEFAULT_MIN_DURATION_MIN),
                    step=5,
                    key=f"min_duration_{source.value}",
                )
                subject_filter = st.text_input("Filter subject (optional)", "")

                if st.button("Refresh night list", key="refresh_night_list", use_container_width=True):
                    st.session_state.nights_cache = {}
                    st.session_state.metadata_cache = {}
                    st.session_state.engines_cache = {}

                catalog = list_nights_with_metadata(
                    s3_refs,
                    source,
                    start_date=start_date,
                    end_date=end_date,
                    min_duration_min=float(min_duration),
                    meta_df=meta_df,
                )
                catalog_entries = catalog
                if subject_filter:
                    catalog_entries = [
                        e for e in catalog_entries if subject_filter in e.ref.subject
                    ]
                all_refs = [entry.ref for entry in catalog_entries]

                bucket = all_refs[0].bucket if all_refs else s3_refs[0].bucket if s3_refs else "—"
                st.caption(
                    f"Showing **{len(all_refs)}** of **{s3_ref_count}** S3 nights "
                    f"(metadata date/duration filters · {bucket})"
                )
                if s3_ref_count and not all_refs:
                    st.warning("No sessions match the selected metadata filters.")
        except Exception as exc:
            st.warning(f"Could not load catalog: {exc}")

    st.divider()
    st.subheader("Parameter sets")
    saved = ParameterSet.list_available()
    if saved:
        pick = st.selectbox("Load preset", ["(custom)"] + [p["name"] for p in saved])
    else:
        pick = "(custom)"
        st.caption("No saved presets in config/parameter_sets/")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Sleep Data Parameter Testbed")
st.markdown(
    "Test **conditioning**, **flatline scoring**, **training windows**, and **evaluation** "
    "on Zephyr and PSG nights"
)
st.markdown("---")

st.header("Configuration")

# --- Step 1: Nights ---
with st.expander("Step 1: Select nights", expanded=True):
    if catalog_entries:
        st.subheader("Nights of sleep")
        st.caption(
            "Session metadata for the current catalog (matches dashboard filters). "
            "Review before selecting a night to analyze."
        )
        st.dataframe(
            catalog_to_display_dataframe(catalog_entries),
            use_container_width=True,
            hide_index=True,
        )
        st.divider()

    mode = st.radio("Mode", ["Single night", "Multiple nights"], horizontal=True)

    if not all_refs:
        st.info(
            "Connect AWS, ensure session metadata exists, and adjust filters "
            "to see nights in the catalog."
        )
        selected_ids: List[str] = []
    elif mode == "Single night":
        choice = st.selectbox(
            "Night",
            options=all_refs,
            format_func=night_label,
        )
        selected_ids = [choice.night_id] if choice else []
    else:
        selected_ids = st.multiselect(
            "Nights",
            options=[r.night_id for r in all_refs],
            default=[all_refs[0].night_id] if all_refs else [],
            format_func=lambda nid: night_label(refs_from_ids(all_refs, [nid])[0]) if refs_from_ids(all_refs, [nid]) else nid,
        )

    st.session_state.selected_night_ids = selected_ids
    if selected_ids:
        st.success(f"{len(selected_ids)} night(s) selected")

# --- Step 2: Subset (optional) ---
with st.expander("Step 2: Time subset (optional)", expanded=False):
    subset_mode = st.radio("Subset", ["Full night", "Custom (minutes)"], horizontal=True)
    subset_str = "full"
    if subset_mode == "Custom (minutes)":
        subset_str = st.text_input("Range (start-end minutes)", "0-60")
    st.caption("Custom range applies to **single-night** runs only.")

# --- Step 3: Model ---
with st.expander("Step 3: Model", expanded=True):
    model_mode = st.radio(
        "Model mode",
        [
            "Scoring only (no model)",
            "Use existing S3 engine (.ptl)",
            "Train new model",
        ],
    )

    model_architecture = "CNN"
    train_epochs = 5
    train_batch_size = 32
    train_learning_rate = 0.001
    early_stopping_patience = 5
    glm_C = 1.0
    glm_max_iter = 1000
    glm_penalty = "l2"
    glm_solver = "lbfgs"
    xgb_n_estimators = 100
    xgb_max_depth = 6
    xgb_learning_rate = 0.1
    xgb_subsample = 0.8
    xgb_colsample_bytree = 0.8
    xgb_min_child_weight = 1.0
    engine_s3_key = None
    auto_match_engine = True

    if model_mode == "Use existing S3 engine (.ptl)":
        auto_match_engine = st.checkbox(
            "Auto-match engine to each night",
            value=True,
            help="Uses engines/{dataset}/{signal}/{night_id}.ptl in the data bucket.",
        )
        if not auto_match_engine and all_refs and aws_configured():
            ds = all_refs[0].dataset
            try:
                engines = fetch_engines(source, ds, signal)
                if engines:
                    picked = st.selectbox(
                        "S3 engine",
                        options=engines,
                        format_func=lambda e: e.s3_key,
                    )
                    engine_s3_key = picked.s3_key
                else:
                    st.warning(f"No .ptl files under engines/{ds}/{signal}/")
            except Exception as exc:
                st.warning(f"Could not list engines: {exc}")
        elif auto_match_engine and selected_ids and aws_configured():
            ref0 = refs_from_ids(all_refs, [selected_ids[0]])[0]
            eng = engine_for_night(ref0)
            if eng:
                st.caption(f"Example match: `{eng.s3_key}`")
            else:
                st.warning(f"No engine for `{ref0.night_id}` — pick a manual engine or train new.")

    elif model_mode == "Train new model":
        saved_model = (
            ParameterSet.load(pick).config.get("model", {})
            if pick != "(custom)" and saved
            else ParameterSet().config.get("model", {})
        )
        saved_arch = MODEL_ARCH_FROM_BACKEND.get(saved_model.get("architecture", "cnn"), "CNN")
        model_architecture = st.selectbox(
            "Architecture",
            MODEL_ARCHITECTURES,
            index=MODEL_ARCHITECTURES.index(saved_arch),
        )

        if model_architecture in ("CNN", "LSTM"):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                train_epochs = st.number_input(
                    "Epochs",
                    1,
                    100,
                    int(saved_model.get("epochs", 5)),
                )
            with c2:
                train_batch_size = st.number_input(
                    "Batch size",
                    8,
                    128,
                    int(saved_model.get("batch_size", 32)),
                    8,
                )
            with c3:
                train_learning_rate = st.select_slider(
                    "Learning rate",
                    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                    value=float(saved_model.get("learning_rate", 0.001)),
                )
            with c4:
                early_stopping_patience = st.number_input(
                    "Early stop patience",
                    1,
                    20,
                    int(saved_model.get("early_stopping_patience", 5)),
                    help="Epochs without val-loss improvement before stopping.",
                )
        elif model_architecture == "GLM":
            st.caption("Multinomial logistic regression on flattened window samples.")
            c1, c2 = st.columns(2)
            with c1:
                glm_C = st.number_input(
                    "C (inverse reg.)",
                    0.001,
                    100.0,
                    float(saved_model.get("glm_C", 1.0)),
                    format="%.4f",
                )
                glm_penalty = st.selectbox(
                    "Penalty",
                    ["l2", "l1", "elasticnet"],
                    index=["l2", "l1", "elasticnet"].index(saved_model.get("glm_penalty", "l2")),
                )
            with c2:
                glm_max_iter = st.number_input(
                    "Max iterations",
                    100,
                    5000,
                    int(saved_model.get("glm_max_iter", 1000)),
                    100,
                )
                if glm_penalty == "elasticnet":
                    glm_solver = "saga"
                    st.caption("Solver: **saga** (required for elasticnet).")
                else:
                    solver_opts = ["lbfgs", "saga"] if glm_penalty == "l2" else ["liblinear", "saga"]
                    default_solver = saved_model.get("glm_solver", "lbfgs")
                    if default_solver not in solver_opts:
                        default_solver = solver_opts[0]
                    glm_solver = st.selectbox("Solver", solver_opts, index=solver_opts.index(default_solver))
        else:
            st.caption("Gradient-boosted trees on flattened window samples.")
            c1, c2, c3 = st.columns(3)
            with c1:
                xgb_n_estimators = st.number_input(
                    "Trees",
                    10,
                    500,
                    int(saved_model.get("xgb_n_estimators", 100)),
                    10,
                )
                xgb_max_depth = st.number_input(
                    "Max depth",
                    1,
                    20,
                    int(saved_model.get("xgb_max_depth", 6)),
                )
            with c2:
                xgb_learning_rate = st.number_input(
                    "Learning rate",
                    0.01,
                    1.0,
                    float(saved_model.get("xgb_learning_rate", 0.1)),
                    0.01,
                    format="%.3f",
                )
                xgb_subsample = st.slider(
                    "Row subsample",
                    0.1,
                    1.0,
                    float(saved_model.get("xgb_subsample", 0.8)),
                    0.05,
                )
            with c3:
                xgb_colsample_bytree = st.slider(
                    "Column subsample",
                    0.1,
                    1.0,
                    float(saved_model.get("xgb_colsample_bytree", 0.8)),
                    0.05,
                )
                xgb_min_child_weight = st.number_input(
                    "Min child weight",
                    0.0,
                    20.0,
                    float(saved_model.get("xgb_min_child_weight", 1.0)),
                    0.5,
                )

        if len(selected_ids) > 1:
            st.info("Multiple nights: windows are pooled for one training run, then evaluated per night.")
        st.caption(
            f"3 classes · window = width × {DEFAULT_SAMPLE_RATE} Hz "
            f"(default 15 s → {15 * DEFAULT_SAMPLE_RATE} samples)"
        )

# --- Step 4: Parameters ---
with st.expander("Step 4: Processing parameters", expanded=True):
    if pick != "(custom)" and saved:
        base = ParameterSet.load(pick)
        cond = base.config.get("conditioning", {})
        score = base.config.get("scoring", {})
        train = base.config.get("training", {})
    else:
        base = ParameterSet()
        cond, score, train = base.config["conditioning"], base.config["scoring"], base.config["training"]
    pred = base.config.get("prediction", {})

    st.markdown("**Conditioning**")
    saved_audio_steps = list(audio_steps_from_config(cond))
    default_step_labels = [AUDIO_STEP_LABELS[s] for s in AUDIO_CONDITIONING_STEP_ORDER if s in saved_audio_steps]
    audio_step_labels = st.multiselect(
        "Audio conditioning steps",
        options=[AUDIO_STEP_LABELS[s] for s in AUDIO_CONDITIONING_STEP_ORDER],
        default=default_step_labels,
        help="Audio only, in order. Zephyr SpO2 artifact cleanup uses Butterworth separately.",
    )
    audio_conditioning_steps = [
        s for s in AUDIO_CONDITIONING_STEP_ORDER if AUDIO_STEP_LABELS[s] in audio_step_labels
    ]
    if "nonlinear" in audio_conditioning_steps and pick == "(custom)":
        st.caption("Nonlinear slope/scale use defaults unless loaded from a preset.")
    spo2_window = st.slider(
        "SpO2 artifact interp (sec)",
        5,
        30,
        int(cond.get("spo2_interpolation_window", 30)),
        help="Zephyr nights: max artifact segment length to interpolate.",
    )

    st.markdown("**Scoring**")
    c1, c2 = st.columns(2)
    with c1:
        variance_threshold = st.number_input("Variance threshold", 0.0, 0.1, float(score.get("variance_threshold", 0.005)), 0.001, format="%.4f")
        window_sec = st.slider(
            "Window width (sec)",
            5,
            30,
            int(score.get("sliding_window_width", 15)),
            help=f"Scales training labels and {DEFAULT_SAMPLE_RATE} Hz sample length (15 s reference).",
        )
    with c2:
        min_apnea_seconds = st.number_input(
            "Min apnea (sec)",
            5,
            30,
            int(score.get("min_apnea_seconds", 10)),
            help="Scoring flatline detector — not the same as window width.",
        )
    _before = int(round(window_sec * 10 / 15))
    st.caption(
        f"→ {window_sec * DEFAULT_SAMPLE_RATE} samples @ {DEFAULT_SAMPLE_RATE} Hz · "
        f"onset {_before}s before / {window_sec - _before}s after · "
        f"non-apnea gap scales with width"
    )
    if model_mode == "Use existing S3 engine (.ptl)" and window_sec != 15:
        st.warning("S3 .ptl engines require 15 s (120 samples). Train new or set width to 15.")

    st.markdown("**Training / validation**")
    c1, c2 = st.columns(2)
    with c1:
        test_frac = st.slider("Val fraction", 0.1, 0.4, float(train.get("test_frac", 0.2)), 0.05)
    with c2:
        split_strategy = st.selectbox(
            "Split strategy",
            ["random", "by_night", "stratified_by_night"],
            index=["random", "by_night", "stratified_by_night"].index(train.get("split_strategy", "random")),
            help="stratified_by_night matches by_night (hold out whole nights).",
        )

    st.markdown("**Model input features**")
    saved_features = train.get("input_features", list(DEFAULT_INPUT_FEATURES))
    wearable_features_ok = source == DataSource.ZEPHYR
    st.checkbox("Audio", value=True, disabled=True, help="Always included.")
    use_heartrate = st.checkbox(
        "Heart rate",
        value="heartrate" in saved_features,
        disabled=not wearable_features_ok
    )
    use_position = st.checkbox(
        "Position",
        value="position" in saved_features,
        disabled=not wearable_features_ok
    )
    if not wearable_features_ok:
        st.caption("Heart rate and position are only available for Zephyr for now.")
    input_features: List[str] = ["audio"]
    if use_heartrate:
        input_features.append("heartrate")
    if use_position:
        input_features.append("position")
    if features_require_retrain(input_features) and model_mode == "Use existing S3 engine (.ptl)":
        st.warning(
            "Non-default features require a new model. S3 engine mode will switch to **Train new model** at run time."
        )

    st.markdown("**Live post-inference**")
    if model_mode == "Scoring only (no model)":
        st.caption("Not used in scoring-only mode.")
    else:
        st.caption("~1 Hz frames; actuation when onset or flatline prob ≥ threshold for N consecutive frames, then cooldown.")
    c1, c2, c3 = st.columns(3)
    with c1:
        activation_threshold = st.slider(
            "Activation threshold",
            0.5,
            0.95,
            float(pred.get("activation_threshold", 0.7)),
            0.05,
        )
    with c2:
        activation_sec = st.slider(
            "Activation (consecutive frames)",
            1,
            10,
            int(pred.get("activation_seconds", pred.get("activation_time", 3))),
        )
    with c3:
        cooldown_sec = st.slider(
            "Cooldown since last actuation (sec)",
            0.0,
            30.0,
            float(pred.get("seconds_since_last_treatment", 5.0)),
            1.0,
        )
    inference_interval_sec = st.slider(
        "Inference interval (sec)",
        0.5,
        2.0,
        float(pred.get("inference_interval_sec", 1.0)),
        0.5,
    )

ui_params = {
    "model_mode": model_mode,
    "model_architecture": model_architecture,
    "engine_s3_key": engine_s3_key,
    "auto_match_engine": auto_match_engine,
    "train_epochs": train_epochs,
    "train_batch_size": train_batch_size,
    "train_learning_rate": train_learning_rate,
    "early_stopping_patience": early_stopping_patience,
    "glm_C": glm_C,
    "glm_max_iter": glm_max_iter,
    "glm_penalty": glm_penalty,
    "glm_solver": glm_solver,
    "xgb_n_estimators": xgb_n_estimators,
    "xgb_max_depth": xgb_max_depth,
    "xgb_learning_rate": xgb_learning_rate,
    "xgb_subsample": xgb_subsample,
    "xgb_colsample_bytree": xgb_colsample_bytree,
    "xgb_min_child_weight": xgb_min_child_weight,
    "spo2_window_sec": spo2_window,
    "audio_conditioning_steps": audio_conditioning_steps,
    "slope_threshold": float(cond.get("slope_threshold", 0.025)),
    "scale_factor_high": float(cond.get("scale_factor_high", 1.0)),
    "scale_factor_low": float(cond.get("scale_factor_low", 0.75)),
    "variance_threshold": variance_threshold,
    "min_apnea_seconds": min_apnea_seconds,
    "window_sec": window_sec,
    "test_frac": test_frac,
    "split_strategy": split_strategy,
    "input_features": input_features,
    "activation_threshold": activation_threshold,
    "activation_sec": activation_sec,
    "cooldown_sec": cooldown_sec,
    "inference_interval_sec": inference_interval_sec,
}
param_set = ui_config_to_parameter_set(ui_params)
model_cfg, _feature_meta = resolve_model_config(
    model_config_from_parameter_set(param_set), param_set
)

st.markdown("---")
if st.button("Run analysis", type="primary", use_container_width=True):
    if not selected_ids:
        st.error("Select at least one night.")
    elif not aws_configured():
        st.error("Configure AWS credentials first.")
    else:
        st.session_state.run_requested = True

# ===========================================================================
# Results (below configuration)
# ===========================================================================
st.markdown("---")
st.header("Results")

# --- Run batch ---
if st.session_state.run_requested:
    st.session_state.run_requested = False
    refs = refs_from_ids(all_refs, selected_ids)
    subset = parse_subset(subset_str) if subset_mode == "Custom (minutes)" else None
    metadata_by_night_id = {
        entry.ref.night_id: entry.metadata
        for entry in catalog_entries
        if entry.ref.night_id in selected_ids
    }

    progress = st.progress(0.0)
    log_lines: List[str] = []

    def _on_checkpoint(msg: str) -> None:
        log_lines.append(msg)
        if log_ph is not None:
            log_ph.code("\n".join(log_lines))

    def _prog(i, total, nid, phase="processing"):
        progress.progress((i + 1) / total if total else 1.0)

    log_ph = None
    with st.status("Running analysis…", expanded=True) as run_status:
        log_ph = st.empty()
        try:
            if len(refs) == 1 and subset:
                payload = {
                    "per_night": [
                        analyze_night(
                            refs[0],
                            param_set,
                            subset=subset,
                            model_config=model_cfg,
                            status_callback=_on_checkpoint,
                        )
                    ],
                    "errors": [],
                    "config": param_set.to_dict(),
                    "n_success": 1,
                    "n_failed": 0,
                }
            else:
                payload = analyze_batch(
                    refs,
                    param_set,
                    progress_callback=_prog,
                    model_config=model_cfg,
                    status_callback=_on_checkpoint,
                )
            payload["subset"] = subset
            payload["metadata_by_night_id"] = metadata_by_night_id
            st.session_state.test_results = payload
            if log_lines:
                run_status.update(label=f"Done — {log_lines[-1]}")
        except Exception as exc:
            st.error(f"Run failed: {exc}")
            run_status.update(label="Run failed", state="error")
    progress.empty()

results = st.session_state.test_results

if not results:
    st.info("Select nights and parameters, then **Run analysis**.")
else:
    st.success(f"Completed {results.get('n_success', 0)} night(s)")
    if results.get("errors"):
        with st.expander("Errors", expanded=True):
            st.dataframe(pd.DataFrame(results["errors"]), hide_index=True)

    nights_data: List[Dict] = results.get("per_night", [])
    tab_over, tab_sleep, tab_signal, tab_metrics, tab_eval, tab_export = st.tabs(
        [
            "Overview",
            "Sleep summary",
            "Signals",
            "Windows & metrics",
            "Expert & efficacy",
            "Export",
        ]
    )

    with tab_over:
        rows = []
        for n in nights_data:
            rows.append({
                "Night": n["night_id"],
                "Source": n["source"],
                "Flatlines": n["n_flatlines"],
                "Windows": n["n_windows"],
                "Detections": n.get("n_detections", "—"),
                "Det. source": DETECTION_SOURCE_LABELS.get(
                    n.get("detection_source"), n.get("detection_source", "—")
                ),
                "Train": n["n_train"],
                "Val": n["n_val"],
                "Expert": "Yes" if n.get("has_expert") else "—",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        if len(nights_data) == 1:
            n0 = nights_data[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean SpO2", f"{n0.get('spo2_metrics', {}).get('mean_spo2', 0):.1f}")
            c2.metric("% time SpO2 < 90", f"{n0.get('spo2_metrics', {}).get('pct_below_90', 0):.1f}%")
            c3.metric("Prevented desats (sim)", n0.get("simulated_efficacy", {}).get("prevented_desaturations", 0))

    with tab_sleep:
        meta_map = results.get("metadata_by_night_id") or {
            entry.ref.night_id: entry.metadata for entry in catalog_entries
        }
        subset_applied = results.get("subset") if len(nights_data) == 1 else None
        per_summary, combined_summary, summary_caption = summaries_for_results(
            nights_data,
            meta_map,
            subset=subset_applied,
        )
        st.caption(summary_caption)
        if results.get("subset") and len(nights_data) > 1:
            st.caption("Time subset was not applied — multi-night runs use full recordings.")
        if combined_summary.empty:
            st.info("No metadata summary available for the analyzed nights.")
        elif len(nights_data) == 1:
            st.dataframe(combined_summary, use_container_width=True, hide_index=True)
        else:
            st.markdown("**Combined**")
            st.dataframe(combined_summary, use_container_width=True, hide_index=True)
            if not per_summary.empty:
                with st.expander("Per night", expanded=False):
                    st.dataframe(per_summary, use_container_width=True, hide_index=True)

    with tab_signal:
        if nights_data:
            pick_night = st.selectbox(
                "Night",
                options=range(len(nights_data)),
                format_func=lambda i: nights_data[i]["night_id"],
            )
            nd = nights_data[pick_night]
            st.caption(SIGNAL_OVERLAY_LEGEND)
            st.plotly_chart(
                plot_signal_preview(nd["conditioned_df"], nd.get("flatline_times"), nd.get("onset_times")),
                width="stretch",
                key="results_signal_chart",
            )

    with tab_metrics:
        if results.get("pooled_training"):
            pt = results["pooled_training"]
            st.markdown("**Pooled training (multi-night)**")
            pm = pt.get("pooled_val_metrics", {})
            if pm.get("accuracy") is not None:
                st.metric("Pooled val accuracy", f"{pm['accuracy']:.3f}")
            st.caption(
                f"Train windows: {pt.get('n_train_windows', '—')} · "
                f"Val windows: {pt.get('n_val_windows', '—')}"
            )
            st.divider()

        if nights_data:
            pick_i = st.selectbox(
                "Night for metrics",
                options=range(len(nights_data)),
                format_func=lambda i: nights_data[i]["night_id"],
                key="metrics_night_pick",
            )
            nd = nights_data[pick_i]
            mi = nd.get("model_info", {})
            if mi.get("mode") and mi.get("mode") != "none":
                st.caption(
                    f"Model: **{mi.get('mode')}** · "
                    f"{mi.get('model_type', 'cnn')} · "
                    f"{mi.get('engine_s3_key', '')}"
                )
            vm = nd.get("val_metrics", {})
            if "accuracy" in vm:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Val accuracy", f"{vm['accuracy']:.3f}")
                c2.metric("F1 macro", f"{vm.get('f1_macro', 0):.3f}")
                c3.metric("Val windows", vm.get("n_val_windows", nd["n_val"]))
                c4.metric("Train windows", nd["n_train"])
                if "confusion_matrix" in vm:
                    labels = ["0", "1", "2"]
                    st.caption(f"Window labels: {WINDOW_CLASS_LEGEND}")
                    st.plotly_chart(
                        plot_confusion_matrix(vm["confusion_matrix"], labels),
                        width="stretch",
                        key="val_confusion_matrix",
                    )
            if vm.get("label_counts"):
                st.plotly_chart(
                    plot_label_distribution(vm["label_counts"]),
                    width="stretch",
                    key="val_label_distribution",
                )
            if vm.get("mode") == "baseline_majority":
                st.caption("Baseline: majority-class predictor on val windows.")

    with tab_eval:
        for nd in nights_data:
            src = nd.get("detection_source", "scoring")
            src_label = DETECTION_SOURCE_LABELS.get(src, src)
            st.markdown(f"**{nd['night_id']}** — {src_label}")
            if src == "live_postprocessing":
                mi = nd.get("model_info", {})
                st.caption(
                    f"Inference frames: {mi.get('n_inference_frames', '—')} · "
                    f"Positive frames: {mi.get('n_positive_frames', '—')} · "
                    f"Actuations: {mi.get('n_actuations', nd.get('n_detections', 0))}"
                )
            if nd.get("expert_eval"):
                ev = nd["expert_eval"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Expert sensitivity", f"{ev.get('sensitivity', 0):.2f}")
                c2.metric("Precision", f"{ev.get('precision', 0):.2f}")
                c3.metric("Extra detections", ev.get("extra_positives", 0))
            else:
                st.caption("No expert annotations (or Zephyr night).")
            eff = nd.get("simulated_efficacy", {})
            st.markdown(
                f"Simulated SpO₂: **{eff.get('prevented_desaturations', 0)}** prevented desats, "
                f"mean improvement **{eff.get('mean_spo2_improvement', 0):.2f}**"
            )
            pp = eff.get("postprocessing") or {}
            if pp:
                st.caption(
                    f"Recovery window ≈ debounce ({pp.get('debounce_sec', '—')}s) + "
                    f"haptic ({pp.get('haptic_duration_ms', '—')}ms) + trailing obs. "
                )
            st.divider()

    with tab_export:
        export_df = pd.DataFrame([
            {
                "night_id": n["night_id"],
                "flatlines": n["n_flatlines"],
                "windows": n["n_windows"],
                "val_accuracy": n.get("val_metrics", {}).get("accuracy"),
                "expert_sensitivity": (n.get("expert_eval") or {}).get("sensitivity"),
                "prevented_desats": n.get("simulated_efficacy", {}).get("prevented_desaturations"),
            }
            for n in nights_data
        ])
        st.download_button(
            "Download summary CSV",
            export_df.to_csv(index=False),
            file_name=f"testbed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download config JSON",
            json.dumps(results.get("config", {}), indent=2),
            file_name=f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.caption("Testbed v0.1 — same night, rough placeholders")
with c2:
    st.caption("Zephyr · SCIDB · MESA")
with c3:
    if st.button("Reset session"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
