'''Script for conditioning + analytical scoring algorithm''' 
import os
import pandas as pd
import numpy as np
import csv
import random
from datetime import datetime, timedelta
import re
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, RobustScaler
# import wandb
import torch
from utils import *
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter, find_peaks, find_peaks_cwt, hilbert, butter, filtfilt
import matplotlib.dates as mdates
import librosa

pd.set_option("display.max_rows", 500)

# Plot
import matplotlib
import matplotlib.pyplot as plt
import plotly
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px

pd.options.plotting.backend = "plotly"

# Stats
from scipy.stats import zscore

# Abbreviations for plotting
ABBREV = {
    "osa": "O",
    "nst_osa": "O",
    "nst_osa_xp": "O",
    "centrala": "C",
    "mixeda": "M",
    "hypopnea": "H",
    "nst_hypopnea": "H",
    "nst_hypopnea_xp": "H",
    "nst_nonapnea": "",
}
    
def resample(data, reindex = True):
    if reindex:
        data["time"] = data["time"].round().astype(int)
    data = data.drop_duplicates(subset=["time"])

    data.index = pd.to_timedelta(data["time"], "S")

    # Separate position (categorical) from continuous signals
    if "position" in data.columns:
        position = data[["position"]].copy()
        continuous = data.drop(["time", "position"], axis=1)
    else:
        position = None
        continuous = data.drop("time", axis=1)

    # Resample continuous data
    continuous_upsampled = continuous.resample('0.125S').interpolate(method='linear')

    # Resample position using forward-fill
    if position is not None:
        position_upsampled = position.resample('0.125S').ffill()
        data_resampled = pd.concat([continuous_upsampled, position_upsampled], axis=1)
    else:
        data_resampled = continuous_upsampled

    data_resampled = data_resampled.reset_index()
    data_resampled["time"] = data_resampled["time"].dt.total_seconds().round(3)

    return data_resampled
    
# Visualize dataframe as plot
def visualize(df, x="time", y="value", show=False, title=''):
    fig = df.plot(x=x, y=y, width=1700, height=600, title=title)
    fig.update_traces(line=dict(color="gray", width=0.5))
    if show:
        fig.show()
    return fig

# Returns true if two segments overlap
def check_overlap(x1, x2, y1, y2):
    return (x1 <= y2 and y1 <= x2) or (y1 <= x2 and x1 <= y2)

def find_snore_candidates(df, sr, env_col="snore_envelope_z",
                          thr_on=2.0, thr_off=1.0,
                          min_width=0.2, max_width=1.0,
                          merge_gap=0.3,
                          check_spectral=True, spec_sr=8000,
                          lf_cut=500, lf_ratio_thr=0.6): #new
    """
    Identify snore-like candidate events from an envelope trace.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'time' and envelope (env_col).
        If 'wave' is present, spectral features will be computed from it.
    sr : float
        Effective sample rate of the envelope trace (Hz).
    env_col : str
        Column in df with normalized envelope (z-score).
    thr_on, thr_off : float
        Hysteresis thresholds for event start/stop.
    min_width, max_width : float
        Allowed event duration in seconds.
    merge_gap : float
        Merge events separated by less than this (s).
    check_spectral : bool
        If True, run a spectral centroid & LF energy check inside each candidate window.
    spec_sr : int
        Resampling rate for spectral features (Hz).
    lf_cut : float
        Cutoff frequency for "low-frequency" band (Hz).
    lf_ratio_thr : float
        Minimum ratio of LF energy required to keep event.

    Returns
    -------
    candidates : list of dict
        Each dict has {"start", "end", "peak", "centroid", "energy_db", "lf_ratio"}
    """

    # TEMPORARY LOCATION

    z = df[env_col].values
    times = df["time"].values

    # Hysteresis thresholding 
    active = False
    events = []
    start_idx = None
    for i, val in enumerate(z):
        if not active and val >= thr_on:
            active = True
            start_idx = i
        elif active and val <= thr_off:
            active = False
            end_idx = i
            events.append((start_idx, end_idx))

    # Duration filter
    min_samples = int(min_width * sr)
    max_samples = int(max_width * sr)
    events = [(s, e) for (s, e) in events if min_samples <= (e - s) <= max_samples]

    # Merge close events
    merged = []
    for s, e in events:
        if not merged:
            merged.append([s, e])
        else:
            if (s - merged[-1][1]) <= int(merge_gap * sr):
                merged[-1][1] = e
            else:
                merged.append([s, e])

    # Spectral + Energy check
    candidates = []
    for s, e in merged:
        seg = z[s:e]
        peak_idx = np.argmax(seg) + s
        t0, t1, t_peak = times[s], times[e], times[peak_idx]

        # # If df["time"] is in seconds → convert to datetime
        # if np.issubdtype(df["time"].dtype, np.number):
        #     t0 = START_TIME + pd.to_timedelta(times[s], unit="s")
        #     t1 = START_TIME + pd.to_timedelta(times[e], unit="s")
        #     t_peak = START_TIME + pd.to_timedelta(times[peak_idx], unit="s")
        # else:  # already datetime
        #     t0, t1, t_peak = times[s], times[e], times[peak_idx]

        centroid, energy_db, lf_ratio = None, None, None
        keep = True

        if "wave" in df.columns:
            # Map envelope indices to waveform (assume aligned)
            raw = df["wave"].values
            raw_seg = raw[s:e].astype(float)

            # Compute energy in dB
            energy = np.sum(raw_seg**2)
            energy_db = 10 * np.log10(energy + 1e-6)

            if check_spectral:
                # Resample segment for feature analysis
                raw_resamp = librosa.resample(raw_seg, orig_sr=sr, target_sr=spec_sr)

                # Spectral centroid
                c = librosa.feature.spectral_centroid(y=raw_resamp, sr=spec_sr)
                centroid = float(np.mean(c))

                # Power spectrum
                S = np.abs(librosa.stft(raw_resamp))**2
                freqs = librosa.fft_frequencies(sr=spec_sr, n_fft=S.shape[0]*2-2)
                psd = np.mean(S, axis=1)

                # LF energy ratio
                lf_energy = np.sum(psd[freqs < lf_cut])
                total_energy = np.sum(psd)
                lf_ratio = lf_energy / (total_energy + 1e-12)

                # Reject if centroid too high or LF ratio too low
                if centroid > 800 or lf_ratio < lf_ratio_thr:
                    keep = False

        if keep:
            candidates.append({
                "start": float(t0),
                "end": float(t1),
                "peak": float(t_peak),
                "centroid": centroid,
                "energy_db": energy_db,
                "lf_ratio": lf_ratio
            })

    return candidates


''' Define Preprocessing module'''
class Preprocess:
    def __init__(self, cfg):

        self.cfg = cfg
        self.apnea_types = []
        if self.cfg.include_apnea:
            self.apnea_types = ["osa", "centrala", "mixeda"]
        if self.cfg.include_hypopnea:
            self.apnea_types += ["hypopnea"]

    # Make df columns lowercase, drop nulls
    def clean_data(self, df):        
        df.columns = df.columns.str.strip().str.lower() # lowercase
        df = df.dropna() # drop NULLs
        df = df.drop_duplicates() # drop duplicate rows
        return df 

    # Get sample rate from df
    def get_sample_rate(self, df):
        sample_rate = int(1 / (df.iloc[1]["time"] - df.iloc[0]["time"]))
        return sample_rate

    # Downsample df by <downsample_step_size>
    def downsample(self, df, downsample_step_size):
        self.cfg.sample_rate = int(
            1 / (df.iloc[1]["time"] - df.iloc[0]["time"])
        )
        print("Sample rate before downsampling:", self.cfg.sample_rate)
        df = df.iloc[::downsample_step_size, :]
        self.cfg.sample_rate = int(
            1 / (df.iloc[1]["time"] - df.iloc[0]["time"])
        )
        print("Sample rate after downsampling:", self.cfg.sample_rate)
        return df

    def median_filter(self, df, window_size):
        print("-------------------Median filtering---------------------")
        df['value'] = df['value'].rolling(window_size).median()
        df['value'] = df['value'].fillna(0)
        return df 
    
    def minmax_normalize(self, df):
        print("-------------------MinMax Normalizing---------------------")
        minim = np.min(df["value"])
        maxim = np.max(df["value"])
        # compress to between [0,1]
        df["value"] = (df["value"] - minim) / (maxim - minim)
        return df 
    
    def nonlinear_scaling(self, df):
        print("-------------------Nonlinear scaling---------------------")
        df["slope"] = df["value"].rolling(window=self.cfg.sample_rate, min_periods=1).apply(lambda x: (x[-1] - x[0]) / 2,  raw=True)
        df["scale_factor"] = np.where(abs(df["slope"]) > self.cfg.slope_threshold, self.cfg.scale_factor_high, self.cfg.scale_factor_low)
        # Scale nonlinearly
        df["value"] *= df["scale_factor"]
        return df 

    def zscore_standardize(self, df):
        print("------------------Standardize with zscore ---------------------")
        df['value'] = zscore(df["value"])
        return df

    def filter(self, df):
        print(f"Using {self.cfg.filter} filter")
        if self.cfg.filter == "savgol":
            # apply a Savitzky-Golay filter
            df["value"] = savgol_filter(df["value"].values, window_length=11, polyorder=3) 
        return df
    
    def envelope(self, df):
        print(f"Extracting envelope")
        analytic_signal = hilbert(df['value'])
        df["value"] = np.abs(analytic_signal)
        return df
    
    def butterworth_filter(self, data, cutoff = 1, order=4, pass_type = "high"):
        print("-------------------SpO2 Butterworth Filter---------------------")
        nyquist = 0.5 * self.cfg.sample_rate   # Nyquist frequency, spo2 sample rate
        normal_cutoff = cutoff / nyquist  # Normalize cutoff frequency
        b, a = butter(order, normal_cutoff, btype=pass_type, analog=False)
        return filtfilt(b, a, data)
    
    def interpolate_segments(self, df, max_sec=5): 
        max_length = max_sec * 8 #8Hz
        mask = df["artifact"].to_numpy()
        spo2_values = df["oxygen"].copy()
        if "heartrate" in df.columns:
            hr_values = df["heartrate"].copy()

        i = 0
        while i < len(mask):
            if mask[i]:  # Found a True segment
                start = i
                while i < len(mask) and mask[i]:  # Find the end of the True segment
                    i += 1
                
                left_idx = start - 1
                right_idx = i
                
                # Heart Rate Interpolation (No max length)
                if "heartrate" in df.columns:
                    if 0 <= left_idx < len(hr_values) and right_idx < len(hr_values):
                        x = [left_idx, right_idx]
                        y = [hr_values.iloc[left_idx], hr_values.iloc[right_idx]]
                        interp_values = np.interp(range(start, i), x, y)
                        hr_values.iloc[start:i] = interp_values
                    else:
                        # can't interpolate, so skip
                        pass 
                
                
                length = i - start
                # SpO2 Interpolation (max_length)
                if length <= max_length:
                    if 0 <= left_idx < len(spo2_values) and right_idx < len(spo2_values):
                        x = [left_idx, right_idx]
                        y = [spo2_values.iloc[left_idx], spo2_values.iloc[right_idx]]
                        interp_values = np.interp(range(start, i), x, y)
                        spo2_values.iloc[start:i] = interp_values
                        df["artifact"].iloc[start:i] = False
                    else:
                        # can't interpolate, so skip
                        pass 

            else:
                i += 1

        df["oxygen"] = spo2_values
        if "heartrate" in df.columns:
            df["heartrate"] = hr_values
        return df




''' Define Analytics module for analytical scoring of apnea''' 
class Analytics():      
    def __init__(self, cfg):

        self.cfg = cfg
        # original unnormalized df
        cfg.data_df.columns = cfg.data_df.columns.str.strip().str.lower()

        self.orig_df = cfg.data_df
        self.df = self.orig_df.copy()

        # set columns
        self.df.columns = ["time", "value", *self.df.columns[2:]]

        try:
            self.total_seconds = self.df.iloc[-1]["time"]
            print("Total seconds: ", self.total_seconds)
        except Exception as e:
            print(f"Unable to get get total seconds of sleep time: {e}")
        # Get total sleep time
        if self.cfg.logger.has_meta:
            self.length_recording_hours = self.cfg.logger.meta["totalsleeptime_h"]
        else:
            hours = round(self.total_seconds / 3600.0, 3)
            self.cfg.logger.log_meta("totalsleeptime_h", hours)
            self.length_recording_hours = round(self.df.iloc[-1]["time"] / 3600, 3)

        # apnea types
        self.apnea_types = []
        if self.cfg.include_apnea:
            self.apnea_types = ["osa", "centrala", "mixeda"]
        if self.cfg.include_hypopnea:
            self.apnea_types += ["hypopnea"]

        print(f"Apnea Types: {self.apnea_types}")

        # initialize counts for onset extraction algorithm
        self.cfg.logger.all_annot = {}  # contains expert annots, and matched nst annots
        self.cfg.logger.extra_pos_annot = {}  # contains nst false pos annots

        self.cfg.logger.counts["num_expert_annot"] = 0
        self.cfg.logger.counts["num_nst_annot"] = 0
        self.cfg.logger.counts["num_matched"] = 0
        self.cfg.logger.counts["num_missed"] = 0
        self.cfg.logger.counts["num_extra_pos"] = 0

        self.min_apnea_length = int(self.cfg.sample_rate) * int(
            self.cfg.min_apnea_seconds
        )
        self.seq_length = int(self.cfg.sample_rate) * (
            self.cfg.seconds_before_apnea + self.cfg.seconds_after_apnea
        )

    # Perform scoring algorithm: analytcally extract apnea onset events 
    def run_onset_extraction(self):
        print("------------Running Onset Extraction------------")
        self.flatline_times = []
        flatline_values = []
        self.onset_times = []
        max_duration = 120

        # rolling window of variance
        self.df["variance"] = (
            self.df["value"]
            .rolling(window=int(self.cfg.sample_rate * 4), min_periods=1) 
            .apply(lambda x: np.var(x), raw=True) 
        )
        # if variance > threshold, set to binary 1, else binary 0
        self.df["binary_var"] = np.where(
            abs(self.df["variance"]) >= self.cfg.variance_threshold, 1, 0
        )

        bin_str = "".join(str(int(x)) for x in self.df["binary_var"].tolist())
  
        if self.cfg.include_apnea:

            # Search positive sequences at least 10 seconds
            # 1 second of high variance, then at least 10 seconds of low variance
            for x in re.finditer(
                r"1{"
                + re.escape(f"{int(self.cfg.sample_rate)}")
                + "}0{"
                + re.escape(f"{self.min_apnea_length}")
                + r",}",
                bin_str,
            ):

                # indexing for entire flatline
                start_flatline_idx, end_flatline_idx = (
                    x.start() + self.cfg.sample_rate,
                    x.end(),
                )
                
                max_end_idx = start_flatline_idx + (max_duration * self.cfg.sample_rate)
                if end_flatline_idx > max_end_idx:
                    end_flatline_idx = max_end_idx

                # captures apnea onset
                start_onset = self.df.iloc[
                    start_flatline_idx
                    - (self.cfg.seconds_before_apnea * self.cfg.sample_rate)
                ]["time"]
                end_onset = self.df.iloc[
                    start_flatline_idx
                    + (self.cfg.seconds_after_apnea * self.cfg.sample_rate)
                ]["time"]
                self.onset_times.append([start_onset, end_onset])

                # captures apnea flatline
                start_flatline = self.df.iloc[start_flatline_idx]["time"]
                end_flatline = self.df.iloc[end_flatline_idx - 1]["time"]
                self.flatline_times.append([start_flatline, end_flatline])

                # get average flatline value from center of onset event
                avg_flatline_idx = int((start_flatline_idx + end_flatline_idx) / 2)
                avg_flatline_value = self.df.iloc[avg_flatline_idx]["value"]
                flatline_values.append(avg_flatline_value)

        # global flatline value
        if flatline_values != []:
            self.flatline_value = np.mean(flatline_values)
        else:
            self.flatline_value = 0

        num_pos_files = len(self.onset_times)
        print(f"Extracted {num_pos_files} total onset events")
        return num_pos_files

    def run_desat_extraction(self, thresholds=[3, 4], min_duration=10, max_duration=90, contam_threshold=0.2):
        """
        Detects SpO2 desaturation events and computes metrics for multiple thresholds in a single pass.

        Returns:
        dict
            {
                threshold_value: (desat_events_df, errors_df)
            }
        """
        print("------------Running SpO2 Desaturation Extraction------------")

        thresholds = sorted(thresholds)
        results = {thr: {"events": [], "errors": []} for thr in thresholds}

        # Pre-clean data
        self.df = self.df.dropna(subset=['oxygen']).reset_index(drop=True)
        time = self.df['time'].values
        spo2 = self.df['oxygen'].values
        artifact = self.df['artifact'].values
        sample_rate = self.cfg.sample_rate
        total_duration_sec = time[-1] - time[0]

        i = 1
        while i < len(spo2) - 1:
            # Local peak detection
            if spo2[i] > spo2[i - 1] and spo2[i] >= spo2[i + 1]:
                local_peak = spo2[i]
                peak_idx = i

                # Look forward until drop >= lowest threshold
                j = i + 1
                while j < len(spo2) and (local_peak - spo2[j]) < thresholds[0]:
                    if spo2[j] > spo2[j - 1]:
                        break  # upward trend; abandon
                    j += 1

                if j < len(spo2) and (local_peak - spo2[j]) >= thresholds[0]:
                    # Found potential desaturation (at least lowest threshold)
                    start_idx = peak_idx
                    true_start_idx = peak_idx
                    lowest_idx = j

                    # Find true dip (lowest point)
                    while j + 1 < len(spo2) and spo2[j + 1] < spo2[j]:
                        j += 1
                        lowest_idx = j

                    # Go forward until recovery
                    end_idx = j
                    while end_idx + 1 < len(spo2) and spo2[end_idx + 1] >= spo2[end_idx]:
                        end_idx += 1
                        # Stop recovery if fully recovered to within lowest threshold
                        if (local_peak - spo2[end_idx]) < thresholds[0]:
                            break

                    event_data = self.df.iloc[true_start_idx:end_idx + 1]
                    duration_sec = (end_idx - true_start_idx) / sample_rate
                    event_min_spo2 = spo2[lowest_idx]
                    event_depth = local_peak - event_min_spo2
                    artifact_proportion = event_data['artifact'].mean()

                    # Evaluate this event for all thresholds
                    for thr in thresholds:
                        reason = None
                        if (local_peak - event_min_spo2) < thr:
                            reason = 'insufficient_drop'
                        elif artifact_proportion > contam_threshold:
                            reason = 'artifact_contamination'
                        elif duration_sec > max_duration:
                            reason = 'too_long'
                        elif duration_sec < min_duration:
                            reason = 'too_short'

                        if reason:
                            results[thr]["errors"].append({
                                'Start': time[true_start_idx],
                                'End': time[end_idx],
                                'Reason': reason,
                                'Duration(s)': duration_sec,
                                'Threshold': thr,
                                'Depth': event_depth,
                                'Min SpO2': event_min_spo2,
                                'SpO2 Peak': local_peak,
                                'Artifact Proportion': artifact_proportion
                            })
                        else:
                            spo2_below_thr = np.maximum(local_peak - event_data['oxygen'], 0)
                            spo2_below_90 = np.maximum(90 - event_data['oxygen'], 0)
                            time_intervals = np.diff(event_data['time'], prepend=event_data['time'].iloc[0])

                            hypoxic_burden_sec = np.sum(spo2_below_thr * time_intervals)
                            hypoxic_burden_90_sec = np.sum(spo2_below_90 * time_intervals)
                            hypoxic_burden_min_hr = (hypoxic_burden_sec / 60) * (3600 / total_duration_sec)
                            hypoxic_burden_90_min_hr = (hypoxic_burden_90_sec / 60) * (3600 / total_duration_sec)

                            results[thr]["events"].append({
                                'Start': time[true_start_idx],
                                'End': time[end_idx],
                                'Duration(s)': duration_sec,
                                'Depth': event_depth,
                                'SpO2 Peak': local_peak,
                                'SpO2 Dip': event_min_spo2,
                                'Hypoxic Burden(%-sec)': round(hypoxic_burden_sec, 3),
                                'Hypoxic Burden(%min/hr)': round(hypoxic_burden_min_hr, 3),
                                'Hypoxic Burden 90(%-sec)': round(hypoxic_burden_90_sec, 3),
                                'Hypoxic Burden 90(%min/hr)': round(hypoxic_burden_90_min_hr, 3),
                                'Threshold': thr
                            })

                    i = end_idx  # skip ahead past this event
            i += 1


        event_columns = [
            'Start', 'End', 'Duration(s)', 'Depth', 'SpO2 Peak', 'SpO2 Dip',
            'Hypoxic Burden(%-sec)', 'Hypoxic Burden(%min/hr)',
            'Hypoxic Burden 90(%-sec)', 'Hypoxic Burden 90(%min/hr)', 'Threshold'
        ]
        error_columns = [
            'Start', 'End', 'Reason', 'Duration(s)', 'Threshold', 'Depth',
            'Min SpO2', 'SpO2 Peak', 'Artifact Proportion'
        ]
        # Convert to DataFrames
        final_results = {}
        for thr, data in results.items():
            desat_df = pd.DataFrame(data["events"])
            errors_df = pd.DataFrame(data["errors"])
            if desat_df.empty:
                desat_df = pd.DataFrame(columns=event_columns)
            if errors_df.empty:
                errors_df = pd.DataFrame(columns=error_columns)
            
            final_results[thr] = (desat_df, errors_df)

        return final_results

    def detect_spo2_below_90(self, min_duration_sec=3, contam_threshold=0.2):
        """
        Detects periods where SpO2 is continuously below 90%, excluding high-artifact segments.
        """
        print("------------Running SpO2 < 90 Detection------------")

        df = self.df.dropna(subset=['oxygen']).reset_index(drop=True)
        spo2 = df['oxygen'].values
        time = df['time'].values
        artifact = df['artifact'].values
        sample_rate = self.cfg.sample_rate
        total_duration_sec = time[-1] - time[0]

        below_90 = spo2 < 90
        events = []
        in_event = False
        start_idx = None

        for i, is_low in enumerate(below_90):
            if is_low and not in_event:
                in_event = True
                start_idx = i
            elif not is_low and in_event:
                end_idx = i - 1
                duration_sec = (end_idx - start_idx + 1) / sample_rate
                if duration_sec >= min_duration_sec:
                    artifact_segment = artifact[start_idx:end_idx + 1]
                    artifact_ratio = artifact_segment.mean()

                    if artifact_ratio <= contam_threshold:
                        event_data = df.iloc[start_idx:end_idx + 1]

                        # Hypoxic burden (below 90%)
                        spo2_below_90 = np.maximum(90 - event_data['oxygen'], 0)
                        time_intervals = np.diff(event_data['time'], prepend=event_data['time'].iloc[0])
                        hypoxic_burden_90_sec = np.sum(spo2_below_90 * time_intervals)
                        hypoxic_burden_90_min_hr = (hypoxic_burden_90_sec / 60) * (3600 / total_duration_sec)

                        events.append({
                            'Start': time[start_idx],
                            'End': time[end_idx],
                            'Duration(s)': duration_sec,
                            'Min SpO2': spo2[start_idx:end_idx + 1].min(),
                            'Artifact Proportion': artifact_ratio,
                            'Hypoxic Burden 90(%-sec)': round(hypoxic_burden_90_sec, 3),
                            'Hypoxic Burden 90(%min/hr)': round(hypoxic_burden_90_min_hr, 3)
                        })
                in_event = False

        # Handle case where signal ends during an event
        if in_event:
            end_idx = len(spo2) - 1
            duration_sec = (end_idx - start_idx + 1) / sample_rate
            if duration_sec >= min_duration_sec:
                artifact_segment = artifact[start_idx:end_idx + 1]
                artifact_ratio = artifact_segment.mean()

                if artifact_ratio <= contam_threshold:
                    event_data = df.iloc[start_idx:end_idx + 1]

                    # Hypoxic burden (below 90%)
                    spo2_below_90 = np.maximum(90 - event_data['oxygen'], 0)
                    time_intervals = np.diff(event_data['time'], prepend=event_data['time'].iloc[0])
                    hypoxic_burden_90_sec = np.sum(spo2_below_90 * time_intervals)
                    hypoxic_burden_90_min_hr = (hypoxic_burden_90_sec / 60) * (3600 / total_duration_sec)

                    events.append({
                        'Start': time[start_idx],
                        'End': time[end_idx],
                        'Duration(s)': duration_sec,
                        'Min SpO2': spo2[start_idx:end_idx + 1].min(),
                        'Artifact Proportion': artifact_ratio,
                        'Hypoxic Burden 90(%-sec)': round(hypoxic_burden_90_sec, 3),
                        'Hypoxic Burden 90(%min/hr)': round(hypoxic_burden_90_min_hr, 3)
                    })

        return pd.DataFrame(events)



    def process_expert_annotations(self):
        print("------------Processing Expert Annotations------------")
        # read expert annotations
        expert_annot = self.cfg.expert_annot_df
        expert_annot.columns = [x.lower() for x in expert_annot.columns]

        self.cfg.logger.counts["num_expert_annot"] = 0

        for i in range(len(expert_annot)):

            start, duration, apnea_type = (
                expert_annot.loc[i, "start"],
                expert_annot.loc[i, "duration"],
                expert_annot.loc[i, "apnea_type"],
            )

            if apnea_type in self.apnea_types:
                self.cfg.logger.counts["num_expert_annot"] += 1
                self.cfg.logger.all_annot[start] = {apnea_type: (start, duration)}


    def process_nst_annotations(self):
        # keep track of which expert annotations have been matched
        expert_matches = set()
        print("------------Processing NST Annotations------------")

        self.cfg.logger.counts["num_nst_annot"] = len(self.onset_times) 
        if self.cfg.logger.counts["num_nst_annot"] == 0:
            print("******** NO APNEA EVENTS DETECTED, EXITING ********")
            self.onset_fig = visualize(self.df, x="time", y="value")
            # self.onset_fig.show()
            return

        if self.cfg.include_apnea:

            for start, end in self.flatline_times:  # detected nst flatline

                duration = end - start

                found_expert_match = False

                for (
                    k,
                    v,
                ) in (
                    self.cfg.logger.all_annot.items()
                ):  # loop through expert annotations

                    # check if any match with expert annotation
                    for apnea_type, (expert_start, expert_duration) in v.items():
                        if apnea_type == "nst_osa":
                            continue
                        expert_end = expert_start + expert_duration

                        # if there's overlap
                        if abs(expert_start - start) <= 10 or check_overlap(
                            expert_start, expert_end, start, end
                        ):
                            self.cfg.logger.all_annot[k]["nst_osa"] = (start, duration)

                            if expert_start not in expert_matches:
                                expert_matches.add(expert_start)
                                self.cfg.logger.counts["num_matched"] += 1

                            found_expert_match = True
                            break

                # no match with expert annotations --> false positive
                if not found_expert_match:
                    self.cfg.logger.counts["num_extra_pos"] += 1
                    self.cfg.logger.extra_pos_annot[start] = {
                        "nst_osa_xp": (start, duration)
                    }

        if self.cfg.include_hypopnea:
            # HYPOPNEA
            for start, end in self.hypopnea_times:
                duration = end - start

                found_expert_match = False

                for (
                    k,
                    v,
                ) in (
                    self.cfg.logger.all_annot.items()
                ):  # loop through expert annotations

                    for apnea_type, (expert_start, expert_duration) in v.items():
                        if apnea_type == "nst_hypopnea" or apnea_type == "nst_osa":
                            continue
                        expert_end = expert_start + expert_duration

                        if (
                            abs(expert_start - start) <= 5
                            or check_overlap(expert_start, expert_end, start, end)
                            and self.cfg.logger.all_annot[k]
                            and not self.cfg.logger.all_annot[k].get("nst_osa")
                        ):
                            self.cfg.logger.all_annot[k]["nst_hypopnea"] = (
                                start,
                                duration,
                            )

                            if expert_start not in expert_matches:
                                expert_matches.add(expert_start)
                                self.cfg.logger.counts["num_matched"] += 1

                            found_expert_match = True
                            break

                # no match with expert annotations --> false positive
                if not found_expert_match:
                    self.cfg.logger.counts["num_extra_pos"] += 1
                    self.cfg.logger.extra_pos_annot[start] = {
                        "nst_hypopnea_xp": (start, duration)
                    }

        # count num missed: if expert detected, but not nst
        # ideally, each item in dict should have a nst match
        for k, v in self.cfg.logger.all_annot.items():
            if (
                len(v.keys()) > 0
                and "nst_osa" not in v.keys()
                and "nst_hypopnea" not in v.keys()
            ):
                self.cfg.logger.counts["num_missed"] += 1


    def plot_extracted_events(self):

        # normalized signal
        self.onset_fig = visualize(self.df, x="time", y="value")
        # original signal
        self.orig_fig = visualize(self.orig_df, x="time", y="value")

        self.spo2_fig = None
        if "spo2" in self.df.columns:
            self.spo2_fig = visualize(self.df, x="time", y="spo2")

        colors = {
            "osa": "crimson",
            "nst_osa": "coral",
            "nst_osa_xp": "hotpink",
            "centrala": "lavender",
            "mixeda": "purple",
            "hypopnea": "darkblue",
            "nst_hypopnea": "cornflowerblue",
            "nst_hypopnea_xp": "skyblue",
            "nst_nonapnea": "green",
        }

        offset = {
            "nst_osa": -0.5,
            "nst_hypopnea": -0.5,
            "nst_hypopnea_xp": -0.5,
            "nst_osa_xp": -0.5,
            "osa": 0.5,
            "centrala": 0.5,
            "mixeda": 0.5,
            "hypopnea": 0.5,
            "nst_nonapnea": 0,
        }

        legend_count = {
            "osa": 0,
            "centrala": 0,
            "mixeda": 0,
            "nst_osa": 0,
            "nst_osa_xp": 0,
            "hypopnea": 0,
            "nst_hypopnea": 0,
            "nst_hypopnea_xp": 0,
            "nst_nonapnea": 0,
        }

        """ -----------------Plot extracted onset events--------------------"""

        annotations = []

        # plot expert + matched nst
        for events in self.cfg.logger.all_annot.values():
            for apnea_type, (start, duration) in events.items():
                legend_count[apnea_type] += 1
                self.onset_fig.add_trace(
                    go.Scatter(
                        x=(start, start + duration),
                        y=[
                            self.flatline_value + offset[apnea_type],
                            self.flatline_value + offset[apnea_type],
                        ],
                        mode="lines",  # +text",
                        name=apnea_type,
                        # text=[None, f'{ABBREV[apnea_type]}: {round(duration,2)}',1],
                        line=go.scatter.Line(color=colors[apnea_type], width=8),
                        showlegend=False,
                        hovertext="DUR:" + str(duration),
                    )
                )

                annotations.append(
                    dict(
                        yref='y2',
                        font=dict(color="white", size=25),
                        x=start + (duration / 2),
                        y=self.flatline_value + offset[apnea_type] + 0.5,
                        text=f"{ABBREV[apnea_type]}",
                        showarrow=False,
                    )
                )
                annotations.append(
                    dict(
                        yref='y2',
                        font=dict(color="white", size=15),
                        x=start + (duration / 2),
                        y=self.flatline_value + offset[apnea_type] + 0.5,
                        text=f"{ABBREV[apnea_type]}",
                        showarrow=False,
                    )
                )

        # plot false pos nst
        for events in self.cfg.logger.extra_pos_annot.values():
            for apnea_type, (start, duration) in events.items():
                legend_count[apnea_type] += 1
                self.onset_fig.add_trace(
                    go.Scatter(
                        x=(start, start + duration),
                        y=[self.flatline_value, self.flatline_value],
                        mode="lines",  # +text",
                        name=apnea_type,
                        line=go.scatter.Line(color=colors[apnea_type], width=8),
                        showlegend=False,
                        hovertext="DUR:" + str(duration),
                    )
                )

                annotations.append(
                    dict(
                        yref='y2',
                        font=dict(color="white", size=25),
                        x=start + (duration / 2),
                        y=self.flatline_value + offset[apnea_type] + 0.5,
                        text=f"{ABBREV[apnea_type]}",  #: {round(duration,2)}',
                        showarrow=False,
                    )
                )

        # add default legend
        for a in colors.keys():
            legend_text = f"{a}: {legend_count[a]}"
            self.onset_fig.add_trace(
                go.Scatter(
                    x=(0, 0),
                    y=[0, 0],
                    mode="lines",
                    line=go.scatter.Line(color=colors[a], width=8),
                    name=legend_text,
                    showlegend=True,
                )
            )

        # names of plots
        if "spo2" in self.df.columns:
            subplot_titles = (
                "Unnormalized",
                "Normalized with apnea annotations",
                "SpO2",
            )
            fig = make_subplots(
                rows=3, cols=1, subplot_titles=subplot_titles, 
                d_xaxes=True, shared_yaxes=True
            )

            for i in range(len(self.orig_fig["data"])):
                fig.add_trace(self.orig_fig["data"][i], row=1, col=1)
            for i in range(len(self.onset_fig["data"])):
                fig.add_trace(self.onset_fig["data"][i], row=2, col=1)
            for i in range(len(self.spo2_fig["data"])):
                fig.add_trace(self.spo2_fig["data"][i], row=3, col=1)
            fig.add_hline(y=90, row=3, col=1)

        else:
            subplot_titles = "Normalized with apnea annotations"
            fig = make_subplots(
                rows=2, cols=1, subplot_titles=subplot_titles, shared_xaxes=True
            )
            for i in range(len(self.orig_fig["data"])):
                fig.add_trace(self.orig_fig["data"][i], row=1, col=1)
            for i in range(len(self.onset_fig["data"])):
                fig.add_trace(self.onset_fig["data"][i], row=2, col=1)

        # make space for explanation / annotation
        fig.update_layout(
            width=1400,
            height=800,
            margin=dict(l=25, r=25, t=150, b=50),
            paper_bgcolor="black",
            annotations=annotations,
            title_text=self.cfg.plot_file,
            template="plotly_dark",
        )
        text = "\n".join(
            [f"{key}({value})" for key, value in self.cfg.logger.metrics.items()]
        )
        fig.add_annotation(
            dict(font=dict(color="yellow", size=13)),
            text=text,
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=1.1,
            y=1,
            bordercolor="yellow",
            borderwidth=1,
        )
        
        mean_val = np.mean(self.orig_df['value'])

        fig.update_layout(
            # yaxis1=dict(range=[mean_val-10, mean_val+10], fixedrange=False),

            yaxis2=dict(range=[-5, 5], fixedrange=False),
            yaxis3=dict(range=[80, 100], fixedrange=False),
            xaxis=dict(rangeslider=dict(visible=True), type="linear", fixedrange=False),
            font=dict(family="Courier New, monospace", size=20, color="white"),
            plot_bgcolor="black"
        )

        # # clear plot dirs if already existss
        init_dir(os.path.dirname(self.cfg.plot_file))
        print("Writing plot file locally to ......", self.cfg.plot_file)
        fig.write_html(self.cfg.plot_file)

        self.plotly_fig = fig
        if self.cfg.plot_locally:
            fig.show()


        # print('Showing fig')
        # fig.show()

        return self.cfg.plot_file, fig

    def write_output_files(self, bucket_name):
        print("------------Writing NST annotations------------")
        nst_annot_path = os.path.join(self.cfg.nst_annotations_dir, self.cfg.dataset, self.cfg.signal)
        init_dir(nst_annot_path)
        print('NST annot path', nst_annot_path)

        with open(f"{self.cfg.nst_annot_file}", "w", newline="\n") as annots:
            # Write header
            fieldnames = ["start", "duration", "apnea_type"]
            writer = csv.DictWriter(annots, fieldnames=fieldnames)
            writer.writeheader()
            # Write each extracted onset event as a row
            for start_time, end_time in self.flatline_times:
                writer.writerow(
                    {
                        "start": "%.3f" % start_time,
                        "duration": "%.3f" % (end_time - start_time),
                        "apnea_type": "nst_osa",
                    }
                )

            for events in self.cfg.logger.extra_pos_annot.values():
                for apnea_type, (start_time, duration) in events.items():
                    writer.writerow(
                        {
                            "start": "%.3f" % start_time,
                            "duration": "%.3f" % duration,
                            "apnea_type": apnea_type,
                        }
                    )

        nst_annots_df = pd.read_csv(self.cfg.nst_annot_file)
        # combined_annots_df = pd.concat([nst_annots_df, self.cfg.expert_annot_df])
        combined_annots_df = nst_annots_df.sort_values(by=["start"])
        # drop duplicate rows
        combined_annots_df.sort_values(by=["apnea_type"], inplace=True)
        combined_annots_df.drop_duplicates(subset="start", keep="last", inplace=True)
        combined_annots_df.sort_values(by=["start"], inplace=True)
        combined_annots_df.to_csv(self.cfg.nst_annot_file, index=False)

        try:
            self.cfg.s3.upload_file(
                self.cfg.nst_annot_file,
                self.cfg.nst_annot_file,
                bucket_name=bucket_name,
            )
        except Exception as e:
            print(f"WARNING: could not upload nst annot file: {e}")

    def log_all_scores(self):

        expert_AHI = float(self.cfg.logger.counts["num_expert_annot"]) / float(
            self.cfg.logger.meta["totalsleeptime_h"]
        )
        nst_AHI = float(self.cfg.logger.counts["num_nst_annot"]) / float(
            self.cfg.logger.meta["totalsleeptime_h"]
        )

        self.cfg.logger.log_scores("expert_AHI", round(expert_AHI, 2))
        self.cfg.logger.log_scores(
            "expert_AHI_severity", map_ahi_to_severity(float(expert_AHI))
        )

        self.cfg.logger.log_scores("nst_AHI", round(nst_AHI, 2))
        self.cfg.logger.log_scores(
            "nst_AHI_severity", map_ahi_to_severity(float(nst_AHI))
        )



    def get_report_snippet(self, fig):
        print("-----------Getting report snippet-----------")

        def rounded(number, round_to=1200):
            return round(number / round_to) * round_to

        round_to = 600  # 5 min total
        # get start times of OSA/hypopnea
        start_times = [x[0] for x in self.flatline_times] 
        start_times = [rounded(x, round_to) for x in start_times]  # nearest start times
        counts = {}
        for i in start_times:
            counts[i] = counts.get(i, 0) + 1

        if counts == {}:  # if no events detected
            # if recording shorter than snippet interval
            if self.total_seconds < round_to:
                self.dense_start, self.dense_end = 0, self.total_seconds
            else:  # else, choose random range within bounds
                self.dense_start = random.randrange(
                    int(self.total_seconds) - round_to + 1
                )
                self.dense_end = self.dense_start + round_to

        else:
            print("Getting densest region for snippet extraction")
            densest_x = max(counts, key=counts.get)
            # dense start, dense end
            self.dense_start, self.dense_end = (
                densest_x - round_to / 2,
                densest_x + round_to / 2,
            )
        print(f"Densest region: {self.dense_start} to {self.dense_end}")

        snippet_fig = fig
        snippet_fig.update_xaxes(range=(int(self.dense_start), int(self.dense_end)))
        # print('Showing snippet for report...')
        # snippet_fig.show()
        return snippet_fig
        

    def get_combined_plot(self, desat_4, desat_3, interp_window, peak_times, energies, flatline_df, actuations_df):
        """Generates a stacked line chart combining Position, SpO₂, and Breathing."""
        try:
            position_df = self.df.loc[:,["time", "x-axis", "y-axis", "z-axis", "position"]]
            position_df = position_df.loc[position_df["time"]%1==0,:]
            spo2_df = self.df.loc[:,["time", "heartrate", "oxygen", "confidence", "status", "artifact"]]
            breathing_df = self.df.loc[:,["time", "value"]]
        except: 
            print("Error: One or more input files are missing.")
            return
        
        print(position_df.head())
        print(spo2_df.head())
        print(breathing_df.head())

        print(f"artifacts: {spo2_df.artifact.mean()}")
        cond_raw = (spo2_df["oxygen"] > 100) | (spo2_df["oxygen"] < 50)
        print(f"spo2 > 100, spo2 < 50: {cond_raw.mean()}")
        
        # Calculate snore candidate energies
        time = breathing_df["time"].values
        # peak_times = []
        # energies = []

        # print(time[0], snore_cand[0]["start"], snore_cand[0]["end"], time[-1])
        # for c in snore_cand:
        #     s = np.searchsorted(time, c["start"])
        #     e = np.searchsorted(time, c["end"])
        #     seg = breathing_df["value"].values[s:e]

        #     # Compute energy in dB
        #     energy = np.sum(seg**2)
        #     db = 10 * np.log10(energy + 1e-6)
            
        #     peak_times.append(c["peak"])
        #     energies.append(db)
        
        # peak_times = np.array(peak_times)
        # energies = np.array(energies)


        # Find recording start time
        try:
            year = int(self.cfg.session[:4])
            month = int(self.cfg.session[4:6])
            day = int(self.cfg.session[6:8])
            hour = int(self.cfg.session[8:10])
            minute = int(self.cfg.session[10:12])
            second = int(self.cfg.session[12:])
        except Exception as e:
            print(e)
            year, month, day, hour, minute, second = 2024, 1, 1, 0, 0, 0
        START_TIME = datetime(year, month, day, hour, minute, second) 
                   
        # Strip spaces and lowercase column names
        for df in [position_df, spo2_df, breathing_df]:
            df.columns = df.columns.str.strip().str.lower()
            df['time'] = START_TIME + pd.to_timedelta(df['time'], unit='s')
            
        # for c in snore_cand:
        #     c["start"] = START_TIME + pd.to_timedelta(c["start"], unit="s")
        #     c["end"]   = START_TIME + pd.to_timedelta(c["end"], unit="s")
        #     c["peak"]  = START_TIME + pd.to_timedelta(c["peak"], unit="s")
        # peak_times = np.array([c["peak"] for c in snore_cand])
        # energies = np.array([10*np.log10(np.sum(breathing_df.loc[
        #     (breathing_df["time"] >= c["start"]) & (breathing_df["time"] <= c["end"]), "value"]**2) + 1e-6)
        #     for c in snore_cand])


        sub90_df = self.detect_spo2_below_90(contam_threshold=0.0)
        if not sub90_df.empty:
            sub90_df["Start"] = START_TIME + pd.to_timedelta(sub90_df['Start'], unit='s')

        # Convert desat event times to datetime
        desat_df_4 = desat_4.copy()
        desat_df_3 = desat_3.copy()
        desat_df_4["Start"] = START_TIME + pd.to_timedelta(desat_df_4['Start'], unit='s')
        desat_df_4["End"] = START_TIME + pd.to_timedelta(desat_df_4['End'], unit='s')
        desat_df_3["Start"] = START_TIME + pd.to_timedelta(desat_df_3['Start'], unit='s')
        
        # If flatline_df is provided, convert its times to datetime
        # if not flatline_df.empty:
        if flatline_df is not None and not flatline_df.empty:
            flatline_df['end'] = flatline_df['start'] + flatline_df['duration']
            flatline_df["start"] = START_TIME + pd.to_timedelta(flatline_df['start'], unit='s')
            flatline_df["end"] = START_TIME + pd.to_timedelta(flatline_df['end'], unit='s')


        # Collect Hypoxic Burdens into Bins/Windows
        bin_width = 1 * 60 #minutes->seconds
        
        if desat_df_4.empty:
            start_datetime = START_TIME
            end_datetime = START_TIME + pd.Timedelta(seconds=bin_width)
        else:
            start_datetime = desat_df_4['Start'].min()
            end_datetime = desat_df_4['End'].max()
            
        bins = pd.date_range(start=start_datetime, end=end_datetime, freq=f'{bin_width}S')
        
        if desat_df_4.empty:
            # Create an empty burden series with correct dt index
            burden_per_bin = pd.Series(0, index=bins[:-1])
        else:
            desat_df_4['time_bin'] = pd.cut(desat_df_4['End'], bins=bins, right=False)
            burden_per_bin = desat_df_4.groupby('time_bin')['Hypoxic Burden(%-sec)'].sum() / 60
            burden_per_bin.index = burden_per_bin.index.map(lambda x: x.left)
        
        
        # Convert categorical positions to numerical values
        position_mapping = {
            "UPRIGHT": 0,
            "PRONE": 1,
            "RIGHT": 2,
            "SUPINE": 3,
            "LEFT": 4,
            np.nan: np.nan
        }
        position_df["position_numeric"] = position_df["position"].map(position_mapping)
        # position_df['position_numeric'] = pd.to_numeric(position_df['position'], errors='coerce')
        del position_mapping[np.nan]
        

        # Create a stacked figure
        if energies is None:
            plot_count = 6
        else:
            plot_count = 7
        fig, axes = plt.subplots(nrows=plot_count, ncols=1, sharex=True, figsize=(15, 8))

        for ax in axes:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        # Plot Position
        masked_values = np.ma.masked_where(position_df['position_numeric'].isna(), position_df['position_numeric'])
        axes[0].plot(position_df["time"], masked_values, linestyle='-', color="purple")
        axes[0].set_ylabel("Position")
        axes[0].set_yticks(list(position_mapping.values()))
        axes[0].set_yticklabels(list(position_mapping.keys()), fontsize=8)
        axes[0].grid(True)

        # Plot Hypoxic Burden (Binned into 5 minute intervals)
        burden_per_bin.index = pd.to_datetime(burden_per_bin.index)
        # bin_durations = burden_per_bin.index.to_series().diff().dt.total_seconds().iloc[1:]
        # bin_durations = pd.Series([0] + bin_durations.tolist(), index=burden_per_bin.index)/86400 #scale bar widths
        burden_per_bin.index = pd.to_datetime(burden_per_bin.index)

        if len(burden_per_bin) > 1:
            bin_durations = burden_per_bin.index.to_series().diff().dt.total_seconds().iloc[1:]
            # prepend 0 to align lengths
            bin_durations = pd.Series([0] + bin_durations.tolist(), index=burden_per_bin.index) / 86400
        else:
            # fallback: single bin or empty
            bin_durations = pd.Series([0] * len(burden_per_bin), index=burden_per_bin.index) / 86400

        axes[1].bar(
            x=burden_per_bin.index, 
            height=burden_per_bin.values,
            width=bin_durations,
            color='teal'
        )
        axes[1].set_ylabel('HB')
        axes[1].set_ylim(0, max(5, burden_per_bin.max()*1.1))
        axes[1].grid(True)


        # Add Desaturation Event Ticks Subplot
        axes[2].set_ylabel("Desats")

        # Draw vertical ticks for 3% and 4% desats
        for df, name, color, y_val in [
            (desat_df_3, "3%", "green", 3),
            (desat_df_4, "4%", "red", 4),
            (sub90_df, "sub90", "cornflowerblue", 5)
        ]:
            if not df.empty:
                axes[2].scatter(
                    df["Start"], 
                    [y_val] * len(df), 
                    color=color, 
                    marker='|', 
                    s=200, 
                    label=name
                )

        axes[2].set_yticks([3, 4, 5])
        axes[2].set_yticklabels(["3%", "4%", "sub90"])
        axes[2].set_ylim(2.5, 5.5)
        axes[2].grid(True)
#         axes[2].legend(loc='upper right', fontsize=8)

        # SpO₂
        # Filter data where SpO₂ is below 90
        cond = (spo2_df["oxygen"] < 90) & (spo2_df["status"] == 3)
        below_90 = spo2_df[cond]
        # Debugging: Print filtered values
        print("SpO₂ values below 90:\n", below_90)
        # Plot SpO₂
        axes[3].plot(spo2_df["time"], spo2_df["oxygen"], linestyle='-', color="blue")
        # Ovelay tick marks (red downward triangles) at SpO₂ < 90
        axes[3].scatter(below_90["time"], below_90["oxygen"], color="red", marker="v", s=80, label="SpO₂ < 90")
        axes[3].set_ylabel("SpO₂(%)")
        y_floor = 60
        axes[3].set_ylim(y_floor, 100)
        axes[3].grid(True)

        # Define custom major ticks for the y-axis of SpO₂
        axes[3].set_yticks(np.arange(y_floor, 101, 10))  # Tick marks every 10%
        if actuations_df is not None and not actuations_df.empty:
            actuations_df['time'] = START_TIME + pd.to_timedelta(actuations_df['time'], unit='s')
            for ts in actuations_df['time']:
                axes[3].axvline(
                    x=ts,
                    color='red',
                    linestyle='--',
                    linewidth=1,
                    alpha=0.7,
                    label='Actuation Event'
                )

        # Plot Heart Rate
        axes[4].plot(spo2_df["time"], spo2_df["heartrate"], linestyle='-', color="green")
        axes[4].set_ylabel("HR")
        axes[4].set_ylim(40, 125)
        axes[4].grid(True)

        if plot_count == 6:
            axes[5].plot(breathing_df["time"], breathing_df["value"], linestyle='-', color="orange")
            axes[5].set_ylabel("Breath")
            axes[5].set_xlabel("Time")
            # Limit breathing outlier warping
            lb = breathing_df['value'].quantile(0.40) - 5
            ub = breathing_df['value'].quantile(0.60) + 5
            # lb = breathing_df['value'].quantile(0.001) - 5
            # ub = breathing_df['value'].quantile(0.999) + 5
            axes[5].set_ylim(lb, ub)
            axes[5].grid(True)
        else:
            # Plot Snore Candidates
            sampled_breathing = breathing_df.iloc[::10]  # every 10th point
            axes[5].plot(sampled_breathing["time"], sampled_breathing["value"], color="orange", alpha=0.4)
            # axes[5].plot(breathing_df["time"].values, breathing_df["value"].values, color="orange", alpha=0.4)
            # bar_width = pd.to_timedelta(1.5, unit="s")
            # bar_width = pd.to_timedelta(1.5, unit="s") / pd.Timedelta(days=1)  # bar width in days
            # axes[5].bar(peak_times, energies, width=bar_width, color="red", alpha=1)
            # axes[5].bar(peak_times, energies,
            #     width=pd.Timedelta(seconds=1),  # <- 1 second wide
            #     color='red', alpha=0.8)
            bar_width = pd.Timedelta(seconds=13) / pd.Timedelta(days=1)  # convert to days
            axes[5].bar(
                peak_times,
                energies,
                width=bar_width,
                color='red',
                alpha=1,
                align='center'
            )

            axes[5].set_ylabel("Snoring(dB)")
            ub = breathing_df["value"].quantile(0.975) + 5
            axes[5].set_ylim(0, ub)
            axes[5].grid(True)

            # Plot Breathing
            axes[6].plot(breathing_df["time"], breathing_df["value"], linestyle='-', color="orange")
            axes[6].set_ylabel("Breath")
            axes[6].set_xlabel("Time")
            # Limit breathing outlier warping
            lb = breathing_df['value'].quantile(0.40) - 5
            ub = breathing_df['value'].quantile(0.60) + 5
            # lb = breathing_df['value'].quantile(0.001) - 5
            # ub = breathing_df['value'].quantile(0.999) + 5
            axes[6].set_ylim(lb, ub)
            axes[6].grid(True)

        # Set x-axis major ticks to every 30 minutes
        axes[plot_count-1].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
        axes[plot_count-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.xticks(rotation=45)

        # Highlight breathing events if provided
        if flatline_df is not None:
            for _, row in flatline_df.iterrows():
                axes[plot_count-1].axvspan(
                    row["start"], row["end"], 
                    color="gray", alpha=0.4
                )

        # Ensure a light grid is applied to all subplots
        for ax in axes:
            ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

        for ax in axes:  # Loop through all charts
            # Set axis labels font to Times New Roman, size 10
            ax.set_xlabel(ax.get_xlabel(), fontsize=11, fontname="Times New Roman")
            ax.set_ylabel(ax.get_ylabel(), fontsize=11, fontname="Times New Roman")

        # Apply font settings to tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(11)
            label.set_fontname("Times New Roman")

        # Adjust layout and save figure
        plt.tight_layout()
#         plt.savefig(CHART_PATH, bbox_inches='tight')
#         plt.close()
        return fig

    def get_interactive_chart(self, desat_df_4, desat_df_3, interp_window, peak_times, energies, flatline_df, actuations_df):
        # Read data
        try:
            position_df = self.df.loc[:,["time", "x-axis", "y-axis", "z-axis", "position"]]
            position_df = position_df.loc[position_df["time"]%1==0,:]
            spo2_df = self.df.loc[:,["time", "heartrate", "oxygen", "confidence", "status"]]
            breathing_df = self.df.loc[:,["time", "value"]]
        except: #ToDo: complex handling of missing file cases
            print("Error: One or more input files are missing.")
            return

        try:
            year = int(self.cfg.session[:4])
            month = int(self.cfg.session[4:6])
            day = int(self.cfg.session[6:8])
            hour = int(self.cfg.session[8:10])
            minute = int(self.cfg.session[10:12])
            second = int(self.cfg.session[12:])
        except Exception as e:
            print("Date not found")
            year, month, day, hour, minute, second = 2024, 1, 1, 0, 0, 0
        START_TIME = datetime(year, month, day, hour, minute, second) 
        
        
        if flatline_df.empty and not desat_df_4.empty:
            filtered_lines = []
            for start_time in flatline_df["start"]:
                desat_times = desat_df_4["Start"].sort_values().reset_index(drop=True)
                # Compare against all desat_df_4["Start"] times
                match = desat_times[
                    (desat_times >= start_time) &
                    (desat_times <= start_time + 10)
                ]

                if not match.empty:
                    filtered_lines.append(start_time)


        # Convert time and strip spaces
        for df in [position_df, spo2_df, breathing_df]:
            df.columns = df.columns.str.strip().str.lower()
            df['time'] = START_TIME + pd.to_timedelta(df['time'], unit='s')

        sub90_df = self.detect_spo2_below_90(contam_threshold=0.0)
        
        if not np.issubdtype(desat_df_3['Start'].dtype, np.datetime64):
            if not sub90_df.empty:
                sub90_df["Start"] = START_TIME + pd.to_timedelta(sub90_df['Start'], unit='s')
            desat_df_4["Start"] = START_TIME + pd.to_timedelta(desat_df_4['Start'], unit='s')
            desat_df_4["End"] = START_TIME + pd.to_timedelta(desat_df_4['End'], unit='s')
            desat_df_3["Start"] = START_TIME + pd.to_timedelta(desat_df_3['Start'], unit='s')

        # Find local minima for SpO2 signal
#         trough_indices = argrelextrema(spo2_df['oxygen'].values, np.less)[0]  # indices of local minima
        spo2_df['oxygen'] = pd.to_numeric(spo2_df['oxygen'], errors='coerce')
        spo2_df = spo2_df.dropna(subset=['oxygen'])
        oxygen_values = spo2_df['oxygen'].values
        raw_troughs = argrelextrema(oxygen_values, np.less, order=4)[0]
       
        threshold = 90
        spo2_df['is_trough'] = False
        spo2_df.loc[raw_troughs, 'is_trough'] = spo2_df.loc[raw_troughs, 'oxygen'] < threshold

        spo2_df['trough_group'] = (spo2_df['is_trough'] & ~spo2_df['is_trough'].shift(1, fill_value=False)).cumsum()
        trough_starts = spo2_df[spo2_df['is_trough']].groupby('trough_group').head(1)
#         trough_starts = trough_starts[trough_starts["oxygen"]<92]

        # for c in snore_cand:
        #     c["start"] = START_TIME + pd.to_timedelta(c["start"], unit="s")
        #     c["end"]   = START_TIME + pd.to_timedelta(c["end"], unit="s")
        #     c["peak"]  = START_TIME + pd.to_timedelta(c["peak"], unit="s")
        # peak_times = np.array([c["peak"] for c in snore_cand])
        # energies = np.array([10*np.log10(np.sum(breathing_df.loc[
        #     (breathing_df["time"] >= c["start"]) & (breathing_df["time"] <= c["end"]), "value"]**2) + 1e-6)
        #     for c in snore_cand])

        # Position mapping
        position_mapping = {
            'UPRIGHT': 0,
            'PRONE': 1,
            'RIGHT': 2,
            'SUPINE': 3,
            'LEFT': 4
        }
        # position_mapping = {
        #     0:0,
        #     3:1,
        #     6:2,
        #     9:3,
        #     12:4
        # }
        position_df['position_numeric'] = position_df['position'].map(position_mapping)
        # position_df['position_numeric'] = pd.to_numeric(position_df['position'], errors='coerce')


        # Plotly subplots
        fig = make_subplots(rows=7, cols=1, vertical_spacing=0.02, shared_xaxes=True)

        # Plot Position
        fig.add_trace(go.Scatter(x=position_df['time'], y=position_df['position_numeric'],
                                 mode='lines', name='Position'), row=1, col=1)
        fig.update_yaxes(tickvals=list(position_mapping.values()), ticktext=list(position_mapping.keys()), row=1, col=1)

        bin_width = 1 * 60

        # Plot Hypoxic Burden (if desat_df is not empty)
        if desat_df_4.empty:
            start_datetime = START_TIME
            end_datetime = START_TIME + pd.Timedelta(seconds=bin_width)
        else:
            start_datetime = desat_df_4['Start'].min()
            end_datetime = desat_df_4['End'].max()

        bins = pd.date_range(start=start_datetime, end=end_datetime, freq=f'{bin_width}S')

        if desat_df_4.empty:
                # Create an empty burden series with correct dt index
            burden_per_bin = pd.Series(0, index=bins[:-1])
        else:
            desat_df_4['time_bin'] = pd.cut(desat_df_4['End'], bins=bins, right=False)
            burden_per_bin = desat_df_4.groupby('time_bin')['Hypoxic Burden(%-sec)'].sum() / 60
            burden_per_bin.index = burden_per_bin.index.map(lambda x: x.left)

        fig.add_trace(go.Bar(x=burden_per_bin.index, y=burden_per_bin.values,
                                 name='Hypoxic Burden', marker_color='purple'), row=2, col=1)
        
        for df, name, color, y_level in [
            (desat_df_3, "3%", "green", 3),
            (desat_df_4, "4%", "red", 4),
            (sub90_df, "sub90", "cornflowerblue", 5)
        ]:
            if not df.empty:
                df = df.round(3)
                if y_level == 5:
                    hover_text = [
                        f"{name} event<br>Start: {row['Start']}<br>SpO2 Peak: 90<br>SpO2 Dip: {row['Min SpO2']}<br>Duration: {row['Duration(s)']}s"
                        for _, row in df.iterrows()
                    ]
                else:
                    hover_text = [
                        f"{name} event<br>Start {row['Start']}<br>SpO2Peak: {row['SpO2 Peak']}<br>SpO2 Dip: {row['SpO2 Dip']}<br>Duration: {row['Duration(s)']}s"
                        for _, row in df.iterrows()
                    ]


                fig.add_trace(
                    go.Scatter(
                        x=df['Start'],
                        y=[y_level] * len(df),
                        mode='markers',
                        marker=dict(
                            symbol='line-ns',
                            size=16,
                            color='rgba(0,0,0,0)',
                            line=dict(
                                color=color,
                                width=2
                            )
                        ),
                        name=name,
                        hoverinfo='text',
                        text=hover_text, 
                        showlegend=False
                    ), row=3, col=1
                )

        # Plot SpO₂
        fig.add_trace(go.Scatter(x=spo2_df['time'], y=spo2_df['oxygen'],
                                 mode='lines', name='SpO₂', line=dict(color='blue')), row=4, col=1)

        # Add tick marks
        fig.add_trace(go.Scatter(x=trough_starts['time'], y=trough_starts['oxygen'],
                                 mode='markers', name='Min',
                                 marker=dict(color='red', size=6, symbol='cross')), row=4, col=1)
        
        fig.add_annotation(
            x=0.99,  # x-coordinate (in relative terms, 0 to 1)
            y=0.95,  # y-coordinate (in relative terms, 0 to 1)
#             xref="x3", 
#             yref="y3", 
            xref="paper",  # Relative positioning
            yref="paper",
            text=f"I={interp_window}s",  # The text you want to display
            showarrow=False,
            font=dict(
                size=10,
                family="Times New Roman",
                color="#000000"
            ),
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            align="right",
            opacity=0.8,
        )


        # Plot Heart Rate
        fig.add_trace(go.Scatter(x=spo2_df['time'], y=spo2_df['heartrate'],
                                 mode='lines', name='Heart Rate', line=dict(color='green')), row=5, col=1)

        # Plot Snore Candidates
        fig.add_trace(
            go.Scatter(
                x=breathing_df['time'],
                y=breathing_df['value'],
                mode='lines',
                name='Breathing',
                line=dict(color='orange', width=1),
                opacity=0.4
            ),
            row=6, col=1
        )
        # snore_hover = [
        #     f"Snore event<br>Time: {t}<br>Energy: {e:.1f} dB"
        #     for t, e in zip(peak_times, energies)
        # ]

        # fig.add_trace(
        #     go.Bar(
        #         x=peak_times,
        #         y=energies,
        #         width=1500,  
        #         marker_color='red',
        #         opacity=1,
        #         name="Snoring (dB)",
        #         hovertext=snore_hover,
        #         hoverinfo="text"
        #     ),
        #     row=6, col=1
        # )
        # fig.add_trace(
        #     go.Scatter(
        #         x=peak_times,
        #         y=energies,
        #         mode="markers",
        #         marker=dict(color="red", size=8, symbol="line-ns"),  # vertical tick
        #         name="Snoring"
        #     ),
        #     row=6, col=1
        # )
        peak_times_dt = pd.to_datetime(peak_times).to_pydatetime()
        # for t, e in zip(peak_times, energies):
        #     fig.add_trace(
        #         go.Scatter(
        #             x=[t, t],         # vertical line at time t
        #             y=[0, e],         # from baseline to energy value
        #             mode="lines",
        #             line=dict(color="red", width=1),
        #             name="Snore",
        #             showlegend=False
        #         ),
        #         row=6, col=1   # adjust to your subplot
        #     )
        xs = []
        ys = []
        if energies is not None:
            for t, e in zip(peak_times_dt, energies):
                xs.extend([t, t, None])  # None breaks line segments
                ys.extend([0, e, None])

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="red", width=1),
                name="Snore",
                showlegend=False
            ),
            row=6, col=1
        )
   
        # Plot Breathing
        fig.add_trace(go.Scatter(x=breathing_df['time'], y=breathing_df['value'],
                                 mode='lines', name='Breathing', line=dict(color='orange')), row=7, col=1)
        
        # Precompute y0 and y1 for the breathing plot
        y0 = breathing_df['value'].min()
        y1 = breathing_df['value'].max()

#         # Compute end times
#         flatline_c = flatline_df.copy()
#         flatline_c['end'] = flatline_c['start'] + flatline_c['duration']
#         flatline_c["start"] = START_TIME + pd.to_timedelta(flatline_c['start'], unit='s')
#         flatline_c["end"] = START_TIME + pd.to_timedelta(flatline_c['end'], unit='s')
        
        shapes = []

        # if not flatline_df:
        if flatline_df is not None and not flatline_df.empty:
            # flatline_df['end'] = flatline_df['start'] + flatline_df['duration']
            flatline_df['end'] = flatline_df['start'] + pd.to_timedelta(flatline_df['duration'], unit='s')
            # flatline_df["start"] = START_TIME + pd.to_timedelta(flatline_df['start'], unit='s')
            # flatline_df["end"] = START_TIME + pd.to_timedelta(flatline_df['end'], unit='s')

            shapes = [
                dict(
                    type="rect",
                    xref="x7",
                    yref="y7",
                    x0=row['start'],
                    x1=row['end'],
                    y0=y0,
                    y1=y1,
                    fillcolor="rgba(50, 50, 50, 0.4)",
                    line=dict(width=0),
                    layer="below"
                )
                for _, row in flatline_df.iterrows()
            ]

            # vertical_lines = [
            #     dict(
            #         type="line",
            #         xref="x4",  # x-axis of subplot row 4
            #         yref="y4",  # y-axis of subplot row 4
            #         x0=ts,
            #         x1=ts,
            #         y0=70,  # match your SpO₂ y-axis lower limit
            #         y1=100,  # match your SpO₂ y-axis upper limit
            #         line=dict(
            #             color="red",
            #             width=1,
            #             dash="dot"  # optional: 'solid', 'dash', 'dot', 'dashdot'
            #         ),
            #         layer="above"
            #     )
            #     for ts in flatline_df["start"]
            # ]
            # shapes.extend(vertical_lines)

        vertical_lines = []
        if actuations_df is not None and not actuations_df.empty:
            for ts in actuations_df['time']:
                vertical_lines.append(
                    dict(
                        type="line",
                        xref="x4",  # x-axis of subplot row 4
                        yref="y4",  # y-axis of subplot row 4
                        x0=ts,
                        x1=ts,
                        y0=70,  # match your SpO₂ y-axis lower limit
                        y1=100,  # match your SpO₂ y-axis upper limit
                        line=dict(
                            color="red",
                            width=1,
                            dash="dot"
                        ),
                        layer="above"
                    )
                )
        if len(vertical_lines):
            shapes = shapes + vertical_lines if shapes else vertical_lines

        if len(shapes):
            fig.update_layout(shapes=shapes)



        # Update layout
        fig.update_layout(
            height=800, 
            hovermode='x unified', 
            font=dict(size=10, family="Times New Roman"), 
            margin=dict(b=120),
            showlegend=False,
            xaxis7=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=5, label="5m", step="minute", stepmode="backward"),
                        dict(count=10, label="10m", step="minute", stepmode="backward"),
                        dict(count=1, label="1hr", step="hour", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True, thickness=0.03),
                type="date",
                showticklabels=True 
            )
        )
        for i in range(1, 7):  # rows 1 to 5
            fig.update_layout({f'xaxis{i}': dict(matches='x7')})
        
        fig.update_yaxes(title_text="Position", row=1, col=1)
        fig.update_yaxes(title_text="HB", row=2, col=1)
        fig.update_yaxes(title_text="Desats", row=3, col=1)
        fig.update_yaxes(title_text="SpO₂", row=4, col=1)
        fig.update_yaxes(title_text="Heart Rate", row=5, col=1)
        fig.update_yaxes(title_text="Snoring (dB)", row=6, col=1)
        fig.update_yaxes(title_text="Breathing", row=7, col=1)
        
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgrey')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgrey')

        fig.update_yaxes(range=[70, 100], row=4, col=1)  # For SpO₂
#         fig.update_yaxes(range=[0, 75], row=2, col=1)  # For HB
        fig.update_yaxes(range=[2.5, 5.5], row=3, col=1)
        fig.update_yaxes(range=[40, 125], row=5, col=1)  # For Heart Rate
        fig.update_xaxes(range=[breathing_df["time"].min(), breathing_df["time"].max()])

        ub1 = breathing_df["value"].quantile(0.975) + 5
        fig.update_yaxes(range=[0, ub1], row=6, col=1)

        lb = breathing_df['value'].quantile(0.40) - 5
        ub = breathing_df['value'].quantile(0.60) + 5
        # lb = breathing_df['value'].quantile(0.001) - 5
        # ub = breathing_df['value'].quantile(0.999) + 5
        fig.update_yaxes(range=[lb, ub], row=7, col=1)
        
        print("Generated Interactive Plot")

        return fig
