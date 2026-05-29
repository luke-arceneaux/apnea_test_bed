''' General Helper Functions '''
import shutil
import re
import pandas as pd
# import soundfile as sf
import sys, os
from scipy.io import wavfile
import scipy.signal as signal
from wave import open as open_wave
import numpy as np 

# Converts wav file to df, adapted to extract the first channel of stereo data
def wav2df_stereo(wav_file, downsample_step_size=1):
    print("---------Converting wav file to df------------")

    # Split basename and extension
    sep_idx = wav_file.rfind(".")
    base, ext = wav_file[:sep_idx], wav_file[sep_idx+1:]
    out_file = base + ".csv"

    # Read input wav file
    waveFile = open_wave(wav_file, "rb")
    nframes = waveFile.getnframes()        # number of frames
    sample_rate = waveFile.getframerate()  # frame rate
    nchannels = waveFile.getnchannels()    # number of channels
    sampwidth = waveFile.getsampwidth()    # sample width
    
    print(f"Channels: {nchannels}")
    print(f"Sample width: {sampwidth}")
    print("Sample rate detected: ", sample_rate)
    wavFrames = waveFile.readframes(nframes)
    waveFile.close()

    # Create data frame 
    if nchannels == 2:
        print("Stereo file detected. Extracting first channel.")
        # Convert byte data to numpy array and reshape it for stereo
        wavFrames = np.frombuffer(wavFrames, dtype=np.int16)
        wavFrames = wavFrames.reshape(-1, nchannels)
        value = wavFrames[:, 0]  # Extract first channel
    else:
        value = np.frombuffer(wavFrames, dtype=np.int16)
    
    time = np.linspace(0, (1 / sample_rate) * len(value), num=len(value))
    data = np.stack([time, value], axis=1)
    df = pd.DataFrame(data=data, columns=["time", "value"])

    # Downsample if specified
    if downsample_step_size > 1:
        print(f"Length of df before downsampling: {len(df)}")
        df = df.iloc[::downsample_step_size, :]
        sample_rate = sample_rate / downsample_step_size
        print(f"New sample_rate after downsampling: {sample_rate}")

    # Convert df to csv file
    df.to_csv(out_file, sep=",", header=True, index=False)
    print(f"Length of df after downsampling: {len(df)}")
    return df, out_file, int(sample_rate)


def wav2df(wav_file, downsample_step_size=1):
    print("---------Converting wav file to df------------")

    # Split basename and extension
    sep_idx = wav_file.rfind(".")
    base, ext = wav_file[:sep_idx], wav_file[sep_idx+1:]
    out_file = base + ".csv"

    # Read input wav file
    waveFile = open_wave(wav_file, "rb")
    nframes = waveFile.getnframes()        # number of frames
    sample_rate = waveFile.getframerate()  # frame rate
    
    print(waveFile.getnchannels())
    print(waveFile.getsampwidth())
    print("Sample rate detected: ", sample_rate)
    wavFrames = waveFile.readframes(nframes)

    # Create data frame 
    value = np.fromstring(wavFrames, dtype=np.int16)
    time = np.linspace(0, (1 / sample_rate) * len(value), num=len(value))
    data = np.stack([time, value], axis=1)
    df = pd.DataFrame(data=data, columns=["time", "value"])

    # Downsample if specified
    if downsample_step_size > 1:
        print(f"Length of df before downsampling: {len(df)}")
        df = df.iloc[::downsample_step_size, :]
        sample_rate = sample_rate / downsample_step_size
        print(f"New sample_rate after downsampling: {sample_rate}")

    # Convert df to csv file
    df.to_csv(out_file, sep=",", header=True, index=False)
    print(f"Length of df after downsampling: {len(df)}")
    return df, out_file, int(sample_rate)

# Converts wav file to csv
def wav2csv(file):
    sample_rate, data = wavfile.read(str(file))
    print('Loaded! Sample rate:', sample_rate)
    wavData = pd.DataFrame(data)
    wavData.columns = ['M']
    wavData.to_csv(str(file[:-4] + ".csv"), mode='w')
    
def wav2dfprocess_stereo(wav_file, downsample_step_size=1): # wav2dfprocess, adapted to extract the first channel of stereo data
    threshold = 0
    window_size = 500
    output_limit = 65000
    AGC_Setpoint = 50
    deadband = 25
    gain_setpoint_increment = 0.01
    noise_floor_threshold = 0

    # Split basename and extension
    sep_idx = wav_file.rfind(".")
    base, ext = wav_file[:sep_idx], wav_file[sep_idx+1:]
    out_file = base + ".csv"

    # Read input wav file
    waveFile = open_wave(wav_file, "rb")
    nframes = waveFile.getnframes()
    num_channels = waveFile.getnchannels()
    sample_rate = waveFile.getframerate()
    print(f"Channels: {num_channels}")
    print(f"Sample width: {waveFile.getsampwidth()}")
    print(f"Sample rate detected: {sample_rate}")
    wavFrames = waveFile.readframes(nframes)
    waveFile.close()

    # Convert byte data to numpy array
    data = np.frombuffer(wavFrames, dtype=np.int16)
    data = data.reshape((nframes, num_channels))
    

    # Extract the first channel
    if num_channels == 2:
        print("Stereo file detected. Extracting first channel.")
        data = data[:, 0]  # Use only the first channel

    # Create the envelope for the first channel
    envelope_data = np.abs(data)
    
    gain_setpoint = 0
    output_sample = 0
    sample_index = 0

    while sample_index < nframes:
        # Find peak within all indexes in the current window
        window_peak = envelope_data[sample_index] 
        for index in range(sample_index, min(sample_index + window_size, nframes)):
            if envelope_data[index] > window_peak:
                window_peak = envelope_data[index]

        # Adjust gain and mask noise floor
        gain = (window_peak * (gain_setpoint / 100))
        output_sample = window_peak + gain

        # AGC Set gain
#         if output_sample < AGC_Setpoint - deadband:
#             gain_setpoint += gain_setpoint_increment
#         if output_sample > AGC_Setpoint + deadband:
#             gain_setpoint -= gain_setpoint_increment      

        # Noise Floor Cutoff
        if output_sample < noise_floor_threshold:
            output_sample = 0

        # Limiter
        if output_sample > output_limit:
            output_sample = output_limit

        # Write peak to all index locations in current window
        for index in range(sample_index, min(sample_index + window_size, nframes)):
            envelope_data[index] = output_sample

        sample_index += window_size
        if sample_index + window_size > nframes:
            for index in range(sample_index, nframes):
                envelope_data[index] = output_sample
            break
    
    # Create time array and stack with envelope data
    time = np.linspace(0, (1 / sample_rate) * len(envelope_data), num=len(envelope_data))
    data = np.stack([time, envelope_data], axis=1)
    df = pd.DataFrame(data=data, columns=["time", "value"])

    # Downsample if specified
    if downsample_step_size > 1:
        print(f"Length of df before downsampling: {len(df)}")
        df = df.iloc[::downsample_step_size, :]
        sample_rate = sample_rate / downsample_step_size
        print(f"New sample_rate after downsampling: {sample_rate}")
    
    # Convert df to csv file
    df.to_csv(out_file, sep=",", header=True, index=False)
    print(f"Length of df after downsampling: {len(df)}")
    return df, out_file, int(sample_rate)

def wav2dfprocess(wav_file, downsample_step_size=1): # wav2df, with preprocess in between
    threshold = 0                  
    window_size = 500              # Size of the window of array indexes to determine peak value to generate a value written to all index locations in that window
    output_limit = 65000           # Amplitudes higher than thsi value will be clipped at this value
    AGC_Setpoint = 50              # Sets ampliture that AGC adjusts the output amplitude to
    deadband = 25                  # AGC feedback loop deadband
    gain_setpoint_increment = 0.01 # AGC feedback loop gain
    noise_floor_threshold = 0      # Amplitudes above this value will be set to zero

    sep_idx = wav_file.rfind(".")
    base, ext = wav_file[:sep_idx], wav_file[sep_idx+1:]
    out_file = base + ".csv"

    # Read input wav file
    waveFile = open_wave(wav_file, "rb")
    nframes = waveFile.getnframes()        # number of frames
    num_channels = waveFile.getnchannels() # number of channels
    sample_rate = waveFile.getframerate()  # frame rate
    print("nchannels: ", waveFile.getnchannels())
    print("sampwidth: ", waveFile.getsampwidth())
    print("Sample rate detected: ", sample_rate)
    wavFrames = waveFile.readframes(nframes)

    data = np.frombuffer(wavFrames, dtype=np.int16)
    # Reshape the data array to separate channels
    data = data.reshape((nframes, num_channels))
    # Create the envelope for each channel
    envelope_data = np.abs(data)
    
    gain_setpoint = 0
    output_sample = 0
    sample_index = 0

    while sample_index < nframes:
        # Find peak within all indexes in current window Channel 0
        window_peak = envelope_data[sample_index,0] 
        for index in range (sample_index,sample_index + window_size):
            if envelope_data[index,0] > window_peak:
                window_peak = envelope_data[index,0]       
            if envelope_data[index,1] > window_peak:
                window_peak = envelope_data[index,1]                         
    
        # Adjust channel 0 gain and mask noise floor
        gain = (window_peak * (gain_setpoint/100))    
        output_sample = window_peak + gain   
        
        # AGC Set gain
        # if output_sample < AGC_Setpoint - deadband:
        #     gain_setpoint += gain_setpoint_increment
        # if output_sample > AGC_Setpoint + deadband:
        #     gain_setpoint -= gain_setpoint_increment      
    
        # Noise Floor Cutoff
        if output_sample < noise_floor_threshold:
            output_sample = 0
    
        # Limiter
        if output_sample > output_limit:
            output_sample = output_limit
            
        # Write peak to all index locations in current window
        for index in range (sample_index,sample_index + window_size):
            envelope_data[index, 0] = output_sample 

        # Repeat for next window position
        # Code here to do last odd sized window
        sample_index += window_size 
        if (sample_index + window_size) > nframes:
            for index in range (sample_index,nframes):
                envelope_data[index, 0] = output_sample 
            break
        
#         write(output_file_name + "PROCESSED"+ ".wav", sample_rate, envelope_data.astype(np.int16))       
        
    value = np.fromstring(envelope_data, dtype=np.int16)
    time = np.linspace(0, (1 / (sample_rate)) * len(value), num=len(value))
    data = np.stack([time, value], axis=1)
    df = pd.DataFrame(data=data, columns=["time", "value"])
    
    if downsample_step_size > 1:
        print(f"Length of df before downsampling: {len(df)}")
        df = df.iloc[::downsample_step_size, :]
        sample_rate = sample_rate / downsample_step_size
        print(f"New sample_rate after downsampling: {sample_rate}")
    
    df.to_csv(out_file, sep=",", header=True, index=False)
    print(f"Length of df after downsampling: {len(df)}")
    return df, out_file, int(sample_rate)

# def wav2_snore_envelope(
#     wav_file,
#     envelope_method="hilbert",      # "hilbert" or "rms"
#     bp_lo=20, bp_hi=1000,           # snore-focused band (Hz)
#     smooth_ms=150,                  # envelope smoothing
#     frame_ms=25, hop_ms=10,         # for RMS method
#     downsample_step_size=1          # optional: applied to the OUTPUT trace
# ):
#     # --- Read WAV (int16) ---
#     with open_wave(wav_file, "rb") as wf:
#         nframes = wf.getnframes()
#         nch = wf.getnchannels()
#         sr = wf.getframerate()
#         sampwidth = wf.getsampwidth()
#         data = np.frombuffer(wf.readframes(nframes), dtype=np.int16).reshape(nframes, nch)

#     # --- Mono ---
#     x = data[:, 0] if data.shape[1] == 1 else data.mean(axis=1)
#     x = x.astype(np.float32) / 32768.0  # scale to [-1, 1]

#     # --- Band-pass 20–1000 Hz ---
#     sos = signal.butter(6, [bp_lo, bp_hi], btype="bandpass", fs=sr, output="sos")
#     x_bp = signal.sosfiltfilt(sos, x)

#     # --- Envelope ---
#     if envelope_method == "hilbert":
#         env = np.abs(signal.hilbert(x_bp))
#         # smooth with moving average
#         win = max(1, int(sr * smooth_ms / 1000))
#         env = np.convolve(env, np.ones(win)/win, mode="same")
#         times = np.arange(len(env)) / sr
#         trace = env
#     else:  # RMS
#         frame = int(sr * frame_ms / 1000)
#         hop = int(sr * hop_ms / 1000)
#         if frame % 2 == 1: frame += 1
#         # frame-wise RMS (vectorized via striding)
#         n = 1 + (len(x_bp) - frame) // hop
#         idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
#         frames = x_bp[idx]
#         rms = np.sqrt((frames**2).mean(axis=1))
#         times = (idx[:, 0] + frame/2) / sr
#         # optional extra smoothing to make a clean plot
#         win = max(1, int((smooth_ms/1000) / (hop/sr)))
#         if win > 1:
#             rms = np.convolve(rms, np.ones(win)/win, mode="same")
#         trace = rms

#     # --- Normalize for easy thresholding/plotting ---
#     mu, sigma = np.mean(trace), np.std(trace) + 1e-8
#     trace_z = (trace - mu) / sigma

#     # --- Optional downsample (for plotting) ---
#     if downsample_step_size > 1:
#         times = times[::downsample_step_size]
#         trace_z = trace_z[::downsample_step_size]

#     df = pd.DataFrame({"time": times, "snore_envelope_z": trace_z})
#     return df, sr

def wav2_snore_envelope(
    wav_file,
    envelope_method="rms",        # default: lighter than Hilbert
    bp_lo=20, bp_hi=1000,         # snore-focused band (Hz)
    smooth_ms=150,                # envelope smoothing
    downsample_step_size=1
):
    # --- Read WAV (int16) ---
    with open_wave(wav_file, "rb") as wf:
        nframes = wf.getnframes()
        nch = wf.getnchannels()
        sr = wf.getframerate()
        if nframes == 0:
            print("WARNING: WAV file has zero frames.")
            return pd.DataFrame(), sr
        data = np.frombuffer(wf.readframes(nframes), dtype=np.int16).reshape(nframes, nch)
        
    # --- Mono ---
    x = data[:, 0] if data.shape[1] == 1 else data.mean(axis=1)
    x = x.astype(np.float32) / 32768.0

    # --- Band-pass filter (one-pass, less memory) ---
    sos = signal.butter(6, [bp_lo, bp_hi], btype="bandpass", fs=sr, output="sos")
    x_bp = signal.sosfilt(sos, x)

    # --- Envelope ---
    if envelope_method == "hilbert":
        # Hilbert is expensive but we can still keep it in single-pass style
        analytic = signal.hilbert(x_bp)
        env = np.abs(analytic)
    else:  # RMS style (lighter)
        win = int(sr * smooth_ms / 1000)
        if win < 1: win = 1
        # square, rolling mean, sqrt
        sq = x_bp**2
        kernel = np.ones(win) / win
        rms = np.sqrt(np.convolve(sq, kernel, mode="same"))
        env = rms

    # --- Normalize ---
    mu, sigma = np.mean(env), np.std(env) + 1e-8
    trace_z = (env - mu) / sigma
    times = np.arange(len(trace_z)) / sr

    # --- Optional downsample ---
    if downsample_step_size > 1:
        times = times[::downsample_step_size]
        trace_z = trace_z[::downsample_step_size]
        sr = sr / downsample_step_size

    df = pd.DataFrame({"time": times, "snore_envelope_z": trace_z})
    return df, int(sr)


# Helper function to create directory (clears existing dir)
def clear_dir(path): 
    if os.path.isdir(path): shutil.rmtree(path)
    if not os.path.isdir(path):
        os.makedirs(path)
    assert(len(os.listdir(path)) == 0)

# Helper function to create directory (won't clear existing dir)
def init_dir(path): 
    if not os.path.isdir(path):
        os.makedirs(path)

# Get Seconds from HH:MM:SS time string
def get_sec_from_hhmmss(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

# Helper function to rename files in directory 
def rename_files(path):
    files = os.listdir(path)
    for f in files:
        src = os.path.join(path, f)
        dst = os.path.join(path, f[:-4])
        os.rename(src, dst) 

# Parses out dataset, signal, subject, sess from filepath
def parse_filepath(filepath):
    # e.g. raw_data/scidb/cannula/scidb_cannula_subj1447_sess1.csv
    path = os.path.basename(filepath)
    path = os.path.splitext(path)[0]
    matches = re.match(r'(.*)_(.*)_subj(.*)_sess(.*)', path)
    dataset, signal, subject, session = matches.groups()
    return dataset, signal, subject, session

# Mapping of AHI number to apnea severity
def map_ahi_to_severity(ahi):
    if ahi < 5:
        return 'NORMAL'
    elif ahi < 15:
        return 'MILD'
    elif ahi < 30:
        return 'MODERATE'
    else:
        return 'SEVERE'