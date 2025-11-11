import re
import numpy as np
import pandas as pd
import soundfile as sf
import opensmile as sm
import matplotlib.pyplot as plt
import librosa
from supplies import *
#from _supplies import *

def extract_egmaps(fname, windows, sr=16000):
    """
    Extract eGeMAPS features from an audio file.

    Parameters
    ----------
    file_path : str
        Path to the audio file.
    sr : int
        Target sampling rate.

    Returns
    -------
    pd.DataFrame
        DataFrame containing eGeMAPS features.
    """
    # load audio (mono, float32) 
    y, sr = sf.read(fname, dtype="float32")
    if y.ndim == 2:
        y = y.mean(axis=1)
    # eGeMAPSv02 LLDs
    smile = sm.Smile(
        feature_set=sm.FeatureSet.eGeMAPSv02,
        feature_level=sm.FeatureLevel.LowLevelDescriptors, #Functionals,
    )
    dfs = []
    for k, (s, e) in enumerate(windows, 1):
        i0, i1 = int(s * sr), int(e * sr)
        seg = y[i0:i1]
        if len(seg) == 0:
            continue
        df_win = smile.process_signal(seg, sr)              # MultiIndex: (start,end) timedeltas
        # shift index to absolute time
        t_off = pd.to_timedelta(s, unit="s")
        start = df_win.index.get_level_values("start") + t_off
        end   = df_win.index.get_level_values("end") + t_off
        df_win.index = pd.MultiIndex.from_arrays([start, end], names=["start", "end"])
        df_win["window_id"] = k
        df_win["win_start_s"] = s
        df_win["win_end_s"] = e
        dfs.append(df_win)
    lld = pd.concat(dfs).sort_index()
    '''keep1 = [
    "loudness_sma3_amean",
    "jitterLocal_sma3nz_amean",
    "shimmerLocaldB_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    'F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
    "slopeV500-1500_sma3nz_amean"]
    

     rename_lld = {
    "F0semitoneFrom27.5Hz_sma3nz": "Pitch_logF0",
    "loudness_sma3_amean": "Loudness",
    "shimmerLocaldB_sma3nz_amean": "Shimmer",
    "jitterLocal_sma3nz_amean": "Jitter",    
    "F0semitoneFrom27.5Hz_sma3nz_amean": "F0_level",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": "F0_var",
    "slopeV500-1500_sma3nz_amean":"Shrillness"}'''
    keep1 = [ 
        'Loudness_sma3', 'slope500-1500_sma3',  
        'jitterLocal_sma3nz', 'shimmerLocaldB_sma3nz', 
        'F0semitoneFrom27.5Hz_sma3nz','HNRdBACF_sma3nz',
        ]
    rename_lld = {
        "Loudness_sma3": "Loudness",
        "shimmerLocaldB_sma3nz": "Amp_var",
        "jitterLocal_sma3nz": "Freq_var",
        "F0semitoneFrom27.5Hz_sma3nz": "Pitch_lvl",           
        "HNRdBACF_sma3nz": "Breathiness",
        "slope500-1500_sma3":"Shrillness", 
        }

    keep1 = [c for c in keep1 if c in lld.columns] #+ ["window_id", "win_start_s", "win_end_s"]
    lld1 = lld[keep1]

    lld1 = lld1.rename(columns={k: v for k, v in rename_lld.items() if k in lld1.columns})
    

    return lld1


def plot_feature_bars1(df, aggregate, row=0, top_n=20, out_csv="feature_stats.csv"):
    """
    Plot bar scores of features from a GeMAPS-like DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Features DataFrame (features in columns).
    row : int
        Row index to visualize (ignored if aggregate=True).
    aggregate : bool
        If True, plot average feature scores across all rows.
    top_n : int
        Number of features to show (sorted by magnitude).
    """
    dfs =[]
    if aggregate:
        values = df.mean()
        title = "Acoustic Feature Importance: Mean Scores"
        print(values)
        dfs.append(values)
        # save df in csv
        dfs = pd.DataFrame(dfs)
        dfs.to_csv("feature_importance.csv", index=False)
        weights = (values / values.abs().sum()) * 100
        weights = values.round(2)
        print(weights)


        '''
        # Normal negative values
        if (values < 0).any():
            s = pd.Series(values)
            weights = (s / s.abs().sum()) * 100
            weights = weights.round(2)
            print(weights.round(2))
        else:      
            # Normalize to sum to 100%
            s = pd.Series(values)
            weights = (s / s.sum()) * 100
            weights = weights.round(2)
            print(weights.round(2))'''
    else:
        values = df.iloc[row]
        title = f"Feature values at row {row} ({df.index[row]})"

    # Normalize to [-1,1] 
    '''vmax = values.abs().max()
    if vmax != 0:
        weights = values / vmax
    else:
        weights = values'''
    # Sort and select top-N
    weights = weights.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(9, 5))
    ax = weights.plot(kind="barh", color="darkred")
     # zero line
    ax.axvline(0, color="black", linewidth=1)

    # Annotate bars with values
    for i, v in enumerate(weights):
        ax.text(v + 0.01 * weights.max(), i, f"{v:.2f}", va="center", color="black")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    
    xmin = min(-1, weights.min() * 1.1)
    xmax = weights.max() * 1.1
    ax.set_xlim(xmin, xmax)
    
    plt.title(title)
    plt.xlabel("Mean importance score")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_feature_bars(df, aggregate=True, row=0, top_n=20, out_csv="feature_stats.csv"):
    """
    If aggregate=True: compute mean & std across rows for each (numeric) feature
    and plot top-N by |mean| with error bars (std). Also saves CSV with mean/std.
    If aggregate=False: plot values for a single row (no error bars).
    """
    # keep only numeric feature columns
    num = df.select_dtypes(include=[np.number])

    if aggregate:
        stats = num.agg(['mean', 'std']).T  # rows=features, cols=['mean','std']
        stats = stats.sort_values(by='mean', key=lambda s: s.abs(), ascending=False).head(top_n)
        stats.to_csv(out_csv, index=True)

        plt.figure(figsize=(9, 5))
        ax = plt.barh(stats.index, stats['mean'], xerr=stats['std'], capsize=3)
        # zero line
        plt.axvline(0, linewidth=1)
        # annotate values
        for i, (m, s) in enumerate(zip(stats['mean'], stats['std'])):
            plt.text(m + 0.01 * stats['mean'].abs().max(), i, f"{m:.2f} ± {s:.2f}", va="center")
        # limits and aesthetics
        xmin = min(-1, (stats['mean'] - stats['std']).min() * 1.1)
        xmax = (stats['mean'] + stats['std']).max() * 1.1
        plt.xlim(xmin, xmax)
        plt.title("Acoustic Features: mean ± std (top-N by |mean|)")
        plt.xlabel("Value")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        return stats  # DataFrame with mean & std

    else:
        values = num.iloc[row]
        values = values.sort_values(key=lambda s: s.abs(), ascending=False).head(top_n)
        plt.figure(figsize=(9, 5))
        ax = values.plot(kind="barh")
        plt.axvline(0, linewidth=1)
        for i, v in enumerate(values):
            plt.text(v + 0.01 * values.abs().max(), i, f"{v:.2f}", va="center")
        xmin = min(-1, values.min() * 1.1)
        xmax = values.max() * 1.1
        plt.xlim(xmin, xmax)
        plt.title(f"Feature values at row {row}")
        plt.xlabel("Value")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        return values
#______________________________________
def Validation(data):
    all_windows = [(0.0, 0.15),(0.15, 0.3),(0.3, 0.45),
               (0.45, 0.6),(0.6, 0.75),(0.75, 0.9),
               (0.9, 1.05),(1.05, 1.2),(1.2, 1.35),
               (1.35, 1.5),(1.5, 1.65),(1.65, 1.8),(1.8, 1.95),(1.95, 2.1)]
      
    print("\n________________________________\n")  
    print("All clip Mean eGMAPS feature extraction in progress...")
    print("Time steps...", all_windows)
    # Extract eGMAPS features
    lld2 = extract_egmaps(data, all_windows, sr=16000)
    print("Plotting random features...")
    print("________________________________________")
    # plot selected acoustic features
    stats = plot_feature_bars(lld2, row=0, aggregate=True, top_n=15)
    print("\n___________All Clip____________\n",stats,"\n_______________________\n")
    visulize_waveform(data, all_windows, "CRP", 3,"Crema-D")

    #_____________ refelect windows on the input spectrogram ______________
    audio_path = data
    SR = 16000            # one sample rate everywhere
    HOP = 240             # 10 ms hop at 16 kHz

    # Generate log-mel with consistent params
    y, _ = librosa.load(audio_path, sr=SR)
    audio = AudioSegment.from_file(data)
    silence_thresh = audio.dBFS - 14  # ~14 dB below average loudness
    nonsilent = detect_nonsilent(audio, min_silence_len=500, silence_thresh=silence_thresh)
    #nonsilent = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-45)

    # Trim silence (keep nonsilent part)
    if nonsilent:
        start, end = nonsilent[0]  # Process the first nonsilent interval
        trimmed_audio = audio[start:end]
    else:
        trimmed_audio = audio 
    # Convert to NumPy array and resample
    samples = np.array(trimmed_audio.get_array_of_samples())
    samples = librosa.util.fix_length(samples, size=int(SR * 2))  # Ensure length matches 3 seconds

    # Generate mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=samples.astype(np.float32),
        sr=SR,
        n_fft=1024,
        win_length=480,
        hop_length=240,
        n_mels=128
    )
    spec = librosa.power_to_db(mel, ref=np.max)        # shape: [128, T]
    spec_tensor = torch.tensor(spec).unsqueeze(0)      # [1, 128, T]

    # random_windows are already seconds -> use directly
    time_windows_sec = all_windows

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spec, sr=SR, hop_length=HOP,
                            cmap='inferno', y_axis='mel', x_axis='time')
    for start_s, end_s in time_windows_sec:
        plt.axvspan(start_s, end_s, color='white', alpha=0.35)
    plt.xlabel("Time (s)"); plt.ylabel("Frequency (mel)")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
    return lld2

