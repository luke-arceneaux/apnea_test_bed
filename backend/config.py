"""Bucket names, path templates, and defaults for the test bed backend."""

from enum import Enum


class DataSource(str, Enum):
    ZEPHYR = "zephyr"
    SCIDB = "scidb"
    MESA = "mesa"


BUCKET_BY_SOURCE = {
    DataSource.ZEPHYR: "zephyrapptestbucket",
    DataSource.SCIDB: "neurostimstore",
    DataSource.MESA: "neurostimstore",
}

# PSG dataset folder names inside neurostimstore raw_data / expert_annotations
PSG_DATASET_FOLDER = {
    DataSource.SCIDB: "scidb",
    DataSource.MESA: "mesa-commercial-use",
}

# Zephyr: zephyr/{stereo|oxygen|gravity}/zephyr_{signal}_subj{S}_sess{T}.ext
ZEPHYR_NAF_SIGNAL = "stereo"
ZEPHYR_SPO2_SIGNAL = "oxygen"
ZEPHYR_GRAVITY_SIGNAL = "gravity"
ZEPHYR_SIGNALS = (ZEPHYR_NAF_SIGNAL, ZEPHYR_SPO2_SIGNAL, ZEPHYR_GRAVITY_SIGNAL)

# PSG (neurostimstore): raw_data/{folder}/{signal}/...
# SCIDB: cannula file includes spo2 column; MESA: Flow + separate spo2 folder
PSG_NAF_SIGNAL = {
    DataSource.SCIDB: "cannula",
    DataSource.MESA: "Flow",
}
PSG_SPO2_SIGNAL = {
    DataSource.SCIDB: None,
    DataSource.MESA: "spo2",
}
# Raw SpO2 column in CSV before normalization to pipeline ``oxygen``
PSG_SPO2_OXYGEN_COLUMN = {
    DataSource.SCIDB: "spo2",  # embedded in cannula CSV
    DataSource.MESA: "value",  # separate spo2/ CSV
}
PSG_SIGNALS = tuple(PSG_NAF_SIGNAL.values())

DEFAULT_SAMPLE_RATE = 8
DEFAULT_WINDOW_SAMPLES = 120  # 15 s @ 8 Hz

# Flatline / training window defaults (match data_pipeline scripts)
DEFAULT_VARIANCE_THRESHOLD = 0.005
DEFAULT_MIN_APNEA_SECONDS = 10
DEFAULT_SEC_BEFORE_ONSET = 10
DEFAULT_SEC_AFTER_ONSET = 5
DEFAULT_FLATLINE_WINDOW_SEC = 15

# Expert comparison
DEFAULT_EXPERT_MATCH_TOLERANCE_SEC = 5.0

# Train/val
DEFAULT_TEST_FRAC = 0.2
