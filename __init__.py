"""
Sleep Data Parameter Testbed
Core library for testing conditioning, scoring, and prediction parameters
"""

from parameters import ParameterSet
from database import Database

from backend import (
    ConditionedNightData,
    Conditioner,
    ConditioningParams,
    DataSource,
    Evaluator,
    TestRunner,
    condition_night,
    load_and_condition_night,
    resample_to_8hz,
    calculate_audio_metrics,
    calculate_metrics,
    calculate_spo2_metrics,
    extract_flatline_times,
    generate_training_windows,
    list_available_nights,
    load_expert_annotations,
    load_model,
    load_night_data,
    split_windows,
)

__version__ = "0.2.0"

__all__ = [
    "ParameterSet",
    "Database",
    "ConditionedNightData",
    "Conditioner",
    "ConditioningParams",
    "DataSource",
    "condition_night",
    "load_and_condition_night",
    "resample_to_8hz",
    "Evaluator",
    "TestRunner",
    "calculate_audio_metrics",
    "calculate_metrics",
    "calculate_spo2_metrics",
    "extract_flatline_times",
    "generate_training_windows",
    "list_available_nights",
    "load_expert_annotations",
    "load_model",
    "load_night_data",
    "split_windows",
]
