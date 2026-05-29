"""Orchestrate test-bed runs: load nights, score, evaluate, persist results."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from parameters import ParameterSet

from backend.config import DataSource
from backend.conditioning import condition_night, ConditioningParams
from backend.data_loader import list_available_nights, load_night_data
from backend.evaluator import Evaluator
from backend.expert_annotations import expert_events_to_records, load_expert_annotations
from backend.scoring import extract_flatline_times
from backend.temp_cleanup import cleanup_run_artifacts
from backend.training_windows import generate_windows_from_night, split_windows, windows_to_arrays
from backend.window_config import scoring_onset_margins, window_generation_params_from_scoring
from database import Database


class TestRunner:
    """
    Run parameter configurations against one or more nights.

    Does not invoke the full data_pipeline queue worker; uses backend modules only.
    """

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()

    def run_job(
        self,
        source: str,
        night_refs: Optional[List[str]] = None,
        params: Optional[ParameterSet] = None,
        split_strategy: str = "random",
        test_frac: float = 0.2,
    ) -> str:
        """
        Execute a job over selected nights.

        Args:
            source: zephyr | scidb | mesa
            night_refs: list of night_id strings; if None, uses first available night
            params: ParameterSet (scoring thresholds read from config)
        """
        job_id = str(uuid.uuid4())
        src = DataSource(source)
        param_set = params or ParameterSet()
        scoring = param_set.config.get("scoring", {})
        cond_params = ConditioningParams.for_source(src, param_set.config)

        available = list_available_nights(src)
        if not available:
            raise RuntimeError(f"No nights found for source {source}")

        if night_refs:
            nights = [n for n in available if n.night_id in night_refs]
        else:
            nights = available[:1]

        config = {
            "job_name": f"testbed_{source}",
            "model": "",
            "params": param_set,
            "nights": [n.night_id for n in nights],
        }
        self.db.create_job(job_id, config)
        self.db.update_job_status(job_id, "running")

        t0 = time.perf_counter()
        all_windows = []
        night_id_per_window: List[str] = []

        try:
            for night_ref in nights:
                night_data = load_night_data(night_ref)
                cond_params.is_csv_audio = night_ref.primary_key.lower().endswith(".csv")
                conditioned = condition_night(night_data, params=cond_params, config=param_set.config)
                sec_before, sec_after = scoring_onset_margins(scoring)
                wparams = window_generation_params_from_scoring(
                    scoring, conditioned.sample_rate, night_ref.source
                )
                flatline_times, _ = extract_flatline_times(
                    conditioned.df,
                    sample_rate=conditioned.sample_rate,
                    variance_threshold=scoring.get("variance_threshold", 0.005),
                    min_apnea_seconds=int(scoring.get("min_apnea_seconds", 10)),
                    seconds_before_apnea=sec_before,
                    seconds_after_apnea=sec_after,
                )
                night_data.audio = conditioned.df
                night_data.sample_rate = conditioned.sample_rate
                night_data.total_seconds = conditioned.total_seconds
                windows = generate_windows_from_night(night_data, flatline_times, params=wparams)
                all_windows.extend(windows)
                night_id_per_window.extend([night_ref.night_id] * len(windows))

            train, val = split_windows(
                all_windows,
                strategy=split_strategy,
                test_frac=test_frac,
                night_ids=night_id_per_window if split_strategy != "random" else None,
            )

            metrics: Dict[str, Any] = {
                "train_windows": len(train),
                "val_windows": len(val),
                "total_windows": len(all_windows),
            }

            if val:
                _, _, val_seqs = windows_to_arrays(val)
                # Placeholder labels from windows for structure test
                val_labels = [w.label for w in val]
                # Without a loaded model, report window counts only
                metrics["val_label_counts"] = {
                    str(l): val_labels.count(l) for l in set(val_labels)
                }

            expert = load_expert_annotations(nights[0]) if nights else None
            if expert is not None:
                events = expert_events_to_records(expert)
                fake_detections = [{"time": e["start"]} for e in events[: min(5, len(events))]]
                metrics["expert_eval_sample"] = Evaluator.eval_expert_annotations(
                    fake_detections, events
                )

            runtime = time.perf_counter() - t0
            self.db.add_result(
                job_id,
                nights[0].night_id if nights else "unknown",
                {"metrics": metrics, "processing_time": runtime, "status": "completed"},
            )
            self.db.update_job_status(
                job_id,
                "completed",
                processed_nights=len(nights),
                runtime_seconds=runtime,
            )
        except Exception as exc:
            self.db.update_job_status(job_id, "failed", error_message=str(exc))
            raise
        finally:
            cleanup_run_artifacts()

        return job_id
