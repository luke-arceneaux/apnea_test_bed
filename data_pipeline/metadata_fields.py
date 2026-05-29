from datetime import date, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import matplotlib.dates as mdates

stats_fields = {
    "total_record_time", "osa_total", "osa_index", "desat4_total", "desat3_total", "hypoxic_total",
    "spo2_min", "spo2_T90_perc", "spo2_T90_min", 
    "position_supine", "position_prone", "position_left", "position_right", "position_upright", "position_non_supine",
    "chart_code"
}

rename_map = {
    "total_record_time": "Duration (min)",
    "osa_total": "NST Count",
    "osa_index": "ODI",
    "desat3_total": "Desat Count (3%)",
    "desat4_total": "Desat Count (4%)",
    "hypoxic_total": "Hypoxic Burden (4%)",
    "spo2_min": "Min SpO2",
    "spo2_T90_perc": "T90_perc",
    "spo2_T90_min": "T90_min",
    "position_supine": "supine_proportion",
    "position_prone": "prone_proportion",
    "position_left": "left_proportion",
    "position_right": "right_proportion",
    "position_upright": "upright_proportion",
    "position_non_supine": "non_supine_proportion",
    "chart_code": "Device_ID"
}
METADATA_KEY = "metadata/session_metadata.csv"
all_metadata_fields = [
    "Subject_ID", "Session_ID", "Date", "Start Time",
    "Duration (min)", "ODI", "NST Count", "Desat Count (3%)", "Desat Count (4%)", "Desat Count (sub 90%)", "Hypoxic Burden (4%)",
    "Min SpO2", "T90_perc", "T90_min",
    "supine_proportion", "prone_proportion", "left_proportion", "right_proportion", "upright_proportion", "non_supine_proportion",
    "PDF Report", "Interactive Report", "SpO2 Snapshot", "version", "Dataset"
]
def load_metadata(cfg, bucket="zephyrapptestbucket", key=METADATA_KEY):
    """
    Load metadata CSV from S3. If it doesn't exist, return empty DF
    with expected columns.
    """
    try:
        df, _ = cfg.s3.get_file_df(key, bucket_name=bucket)
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        # First run or missing file
        return pd.DataFrame(columns=all_metadata_fields)  # from metadata_fields.py


def save_metadata(cfg, df, bucket="zephyrapptestbucket", key=METADATA_KEY):
    """
    Upload metadata CSV back to S3 (atomic overwrite).
    """
    cfg.s3.upload_df(df, key, bucket_name=bucket)


def generate_admin_fields(subject, session):
    # Calc date and time
    try:
        year = int(session[:4])
        month = int(session[4:6])
        day = int(session[6:8])
        hour = int(session[8:10])
        minute = int(session[10:12])
        second = int(session[12:])
        recording_date = date(year, month, day)
        start_time = time(hour, minute, second)
    except Exception as e:
        print(e)
        year = 2024
        month = 1
        day = 1
        hour = 0
        minute = 0
        second = 0
        recording_date = date(year, month, day)
        start_time = time(hour, minute, second)

    admin_fields = {
        "Subject_ID": str(subject),
        "Session_ID": str(session),
        "Date": recording_date,
        "Start Time": start_time,
    }
    return admin_fields

def rename_fields(dict, rename_map=rename_map):
    renamed_dict = {
        rename_map.get(k, k): v
        for k, v in dict.items()
    }
    return renamed_dict

def generate_spo2_snapshot(spo2_df, output_path, cfg=None, bucket="zephyrapptestbucket"):

    try:
        year = int(cfg.session[:4])
        month = int(cfg.session[4:6])
        day = int(cfg.session[6:8])
        hour = int(cfg.session[8:10])
        minute = int(cfg.session[10:12])
        second = int(cfg.session[12:])
    except Exception:
        year, month, day, hour, minute, second = 2024, 1, 1, 0, 0, 0

    START_TIME = datetime(year, month, day, hour, minute, second)

    # Convert seconds to datetime
    spo2_df["datetime"] = START_TIME + pd.to_timedelta(spo2_df["time"], unit="s")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 1.5))

    below_thresh = spo2_df[spo2_df["oxygen"] < 90]

    ax.plot(spo2_df["datetime"], spo2_df["oxygen"], linewidth=1)
    ax.scatter(
        below_thresh["datetime"],
        below_thresh["oxygen"],
        marker="v",
        s=60,
        color="red",
    )

    ax.set_ylabel("SpO₂ (%)", fontsize=8)
    ax.set_ylim(60, 100)
    ax.grid(True)
    # Auto format based on time range
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator())
    fig.autofmt_xdate()

    plt.tight_layout()

    plt.savefig(output_path, dpi=250)
    plt.close(fig)

    if cfg:
        try:
            cfg.s3.upload_file(output_path, output_path, bucket_name=bucket)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    return "Snapshot saved"

def generate_file_field_paths(cfg, bucket, snapshot_status):
    s3_front = f"s3://{bucket}/"
    report_path = f"reports/{cfg.dataset}/{cfg.signal}/"
    snapshot_path = f"snapshots/{cfg.dataset}/{cfg.signal}/"
    filename = cfg.file_format_data
    report_files = {
        "PDF Report": f"{s3_front}{report_path}{filename}.pdf",
        "Interactive Report": f"{s3_front}{report_path}{filename}.html",
    }
    if snapshot_status is not None:
        report_files["SpO2 Snapshot"] = f"{s3_front}{snapshot_path}{filename}.png"
    else:
        report_files["SpO2 Snapshot"] = "Null"
    return report_files

# missing; admin, paths, sub90
# call outside gen_aux path: calc_sub90, generate spo2 plot

def upsert_row(df, row_dict, id_cols):
    new_df = pd.DataFrame([row_dict]).set_index(id_cols)
    df = df.set_index(id_cols)

    df.update(new_df)
    df = pd.concat([df, new_df[~new_df.index.isin(df.index)]])

    return df.reset_index()

def set_version(dict, version_str):
    dict["version"] = version_str
    print(f"Set metadata version to {version_str}")
    return dict

def set_dataset(dict, dataset_str):
    dict["Dataset"] = dataset_str
    print(f"Set metadata dataset to {dataset_str}")
    return dict

def generate_aux_fields(cfg, bucket, spo2_status):
    print(f"Generating auxiliary fields for subject {cfg.subject}, session {cfg.session} with SpO2 snapshot status: {spo2_status}")
    admin_dict = generate_admin_fields(cfg.subject, cfg.session)
    paths_dict = generate_file_field_paths(cfg, bucket, spo2_status)
    aux_dict = {**admin_dict, **paths_dict}
    return aux_dict
