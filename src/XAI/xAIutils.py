# Standard libraries
import os
import sys
import math
import random
import glob
import csv
from pathlib import Path
from IPython.display import Audio 
# Data handling and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Audio processing
import librosa
import librosa.display
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import pyloudnorm as pyln
import librosa.display
# IPython utilities for audio
from IPython.display import Audio
# To play audio files
import IPython.display as ipd


#
import soundfile as sf
import opensmile as osm

# Progress bar
from tqdm import tqdm

# Scikit-learn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from captum.attr import LayerGradCam
from captum.attr import LayerGradCam, LayerAttribution
from captum.attr import Occlusion

import matplotlib.pyplot as plt
import librosa
import numpy as np
from datetime import datetime
import parselmouth  # for jitter & shimmer
from parselmouth.praat import call



from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names

from zennit.composites import EpsilonPlusFlat
from zennit.canonizers import SequentialMergeBatchNorm

from crp.visualization import FeatureVisualization
from crp.image import plot_grid, imgify
from zennit.torchvision import ResNetCanonizer
from crp.helper import get_layer_names
#from heatmap_proc import sliding_win, overlay_top_windows, get_log_melspe


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def process_dataset(data_path, data_setname, pickle_file, label_column, sample_rate=16000, batch_size=2):
    """
    Generalized dataset processing function.
    
    Args:
        data_path (str): Path to the dataset.
        data_setname (str): Name of the dataset.
        pickle_file (str): Name of the pickle file containing the dataset DataFrame.
        label_column (str): Name of the column containing emotion labels.
        sample_rate (int): Audio sample rate for processing.
        batch_size (int): Batch size for DataLoader.
        
    Returns:
        tuple: ds_train, ds_test, ds_val, dl_train, dl_test, dl_val, unique_labels
    """
    df = pd.read_pickle(pickle_file)
    X = df.iloc[:, 1:].values
    Y = df[label_column].values

    # Split data
    x_train_v, x_test, y_train_v, y_test = train_test_split(X, Y, random_state=42, test_size=0.1, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train_v, y_train_v, random_state=42, test_size=0.1, shuffle=True)

    # Dataset class
    class CustomDataset(Dataset):
        def __init__(self, data, labels, transform=None):
            self.data = data
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            _, spectrogram = self.data[idx]
            label = self.labels[idx]
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
            label = torch.tensor(label, dtype=torch.long)
            if self.transform:
                spectrogram = self.transform(spectrogram)
            return spectrogram, label

    # Create datasets and dataloaders
    ds_train = CustomDataset(x_train, y_train)
    ds_test = CustomDataset(x_test, y_test)
    ds_val = CustomDataset(x_val, y_val)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True)

    unique_labels = df[label_column].unique().tolist()
    return ds_train, ds_test, ds_val, dl_train, dl_test, dl_val, unique_labels

#__________________________________________________

# Model utility functions
def set_parameter_requires_grad(model, feature_extracting, trainable_layers):
    if feature_extracting:
        for name, param in model.named_parameters():
            print(name)
            if name not in trainable_layers:
                param.requires_grad = False
# Model definition
class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False) if in_ch != out_ch else None

        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut:
            identity = self.shortcut(identity)
        out += identity
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, class_num):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = ResNetBlock(32, 64, stride=1)
        self.layer2 = ResNetBlock(64, 128, stride=2)

        # Increased dropout rate
        self.dropout = nn.Dropout(0.50)
        self.fc = nn.Linear(128, class_num)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.layer1(x)
        x = self.layer2(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
    

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates the model on a given dataloader.
    """
    model.eval()  # Set model to evaluation mode
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


# Get the prediction of the model
def get_log_melspe(data, sr=16000):
    # Load audio using pydub
    audio = AudioSegment.from_file(data)
    silence_thresh = audio.dBFS - 14  # ~14 dB below average loudness
    nonsilent = detect_nonsilent(audio, min_silence_len=500, silence_thresh=silence_thresh)
    #nonsilent = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-45)

    # Trim silence (keep nonsilent part)
    if nonsilent:
        start, end = nonsilent[0]  # Process the first nonsilent interval
        trimmed_audio = audio[start:end]
    else:
        trimmed_audio = audio  # Use full audio if no nonsilent parts detected

    # Convert to NumPy array and resample
    samples = np.array(trimmed_audio.get_array_of_samples())
    samples = librosa.util.fix_length(samples, size=int(sr * 2))  # Ensure length matches 3 seconds

    # Generate mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=samples.astype(np.float32),
        sr=sr,
        n_fft=2048,
        win_length=480,
        hop_length=240,
        n_mels=32
    )

    return librosa.power_to_db(mel, ref=np.max)

def plot_log_melspe(data, sr=16000):
    """ Plot the log-mel spectrogram of the audio data. """
    log_mel = get_log_melspe(data, sr)

    #plot the log-mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel, sr=sr, hop_length=240, x_axis='time', y_axis='hz', cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    x_label = 'Time (s)'
    y_label = 'Frequency (Hz)'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def get_prediction(model, data, sr=16000):
    log_mel = get_log_melspe(data)
    log_mel = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0)  # shape [1, 1, 128, T]

    # Predict
    with torch.no_grad():
        log_mel = log_mel.to(device).float()
        output = model(log_mel)
        pred = torch.argmax(output, dim=1).item()

    return pred

############ Audio Segmentation and Processing ############
def sliding_win(input_tensor, win_wid, k, threshold, xai_method):
    '''
    Args:
      input_tensor (torch.Tensor): shape (1, height, width), e.g., torch.Size([1, 128, 201])
      win_wid (int): width of the sliding window.
      k (int): Number of top windows to extract.
      threshold (float): Minimum value to retain in Grad-CAM (mimics red-vs-blue).

    Returns:
      top_k_indices (list): Indices of the top k windows.
      top_windows (list): Spectrogram sections corresponding to top k windows.
      ts (list): Time segments (start, end) of the top windows.
    '''
    if isinstance(input_tensor, np.ndarray):
        input_tensor = torch.from_numpy(input_tensor)

    if input_tensor.ndim == 2:
        input_tensor = input_tensor.unsqueeze(0)  # add channel dim

    _, freq, time = input_tensor.shape

    win_sums = []
    start_indices = []

    # Zero out low-importance regions (blue)
    
    masked_tensor = input_tensor.clone()
    masked_tensor[masked_tensor < threshold] = 0.0

    # Slide window across time
    for i in range(0, time, win_wid):
        window = masked_tensor[:, :, i:i+win_wid]  # Extract window
        window_sum = torch.sum(window, dim=(1, 2))  # Sum importance
        win_sums.append(window_sum.item())
        start_indices.append(i)

    # Top-k windows
    window_sums_tensor = torch.tensor(win_sums)
    top_k_indices = torch.topk(window_sums_tensor, k=k).indices.tolist()
    top_windows = [input_tensor[:, :, start_indices[idx]:start_indices[idx] + win_wid] for idx in top_k_indices]
    ts = [(start_indices[idx], start_indices[idx] + win_wid) for idx in top_k_indices]

    return top_k_indices, top_windows, ts

################XAI Methods######################
#________________________________________________
def CRP_xai(model, data, ima, LABEL_DICT, dataname, label, device):

    # Put model in eval mode
    model.eval()
    log_mel = get_log_melspe(data)
    log_mel = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0)  
    log_mel = log_mel.to(device).float()
    # Forward input
    output = model(log_mel)
    predicted_class = output.argmax(dim=1).item()


    cc = ChannelConcept()
    # mask channel 0 and 2 in batch 0
    mask_fn = cc.mask(0, [0, 2])
    speech = data #ds_test.data[index][0]

    canonizers = [ResNetCanonizer()]
    composite = EpsilonPlusFlat(canonizers)
    # describes how the attributions should be computed on the dataset
    attribution = CondAttribution(model, no_param_grad=True)

    #conditions = [{"y": label}]

    sample = torch.tensor(ima, dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0).to(device) #  add batch and channel dimensions

    #attr = attribution(sample, conditions, composite, mask_map=cc.mask)

    layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])

    softmax = torch.nn.Softmax(dim=-1)
    def select_max(pred):
        id = softmax(pred).argmax(-1).item()
        print(f"wrt. class {id}")
        mask = torch.zeros_like(pred)
        mask[0, id] = pred[0, id]
        return mask

    conditions = [{"layer1.conv1": [35]}] # 35
    heatmap2, _, _, _ = attribution(sample, conditions, composite, init_rel=select_max)
    imgify(heatmap2, symmetric=True)
    if LABEL_DICT[predicted_class] == label:
        pred = True
    else:
        pred = False
    _,_,_,_,_, time_steps = saliency_map_processing(heatmap2, predicted_class, LABEL_DICT, "CRP", dataname, data)

    visulize_waveform(data, time_steps, "CRP", label,dataname)
    return time_steps



#________________________________________________
def GradCAM(model, data, LABEL_DICT, dataname, label, device):
    '''
       # GradCAM for SER
       Ref.Kim & Kwak (2024), Otsuki et al. (2020),Usha & Alex (2024))

    '''
    # Put model in eval mode
    model.eval()
    log_mel = get_log_melspe(data)
    plot_log_melspe(data, sr=16000)
    log_mel = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0)  
    log_mel = log_mel.to(device).float()
    # Forward input
    output = model(log_mel)
    predicted_class = output.argmax(dim=1).item()

    # Grad-CAM on the last conv layer
    gradcam = LayerGradCam(model, model.layer2)
    attributions = gradcam.attribute(log_mel, target=predicted_class)

    # Upsample to input size (128x128 or similar)
    upsampled_attr = LayerAttribution.interpolate(attributions, log_mel.shape[2:])
    gradcamp_array = upsampled_attr.cpu().detach().numpy()
    gradcamp_array = np.squeeze(gradcamp_array)  # Remove batch and channel dimensions

    # Visualize
    sr = 16000  # Sample rate
    n_mels = gradcamp_array.shape[0]  # Number of mel bands
    hop_length = 240  # Hop length used in mel spectrogram generation

    # x-axis time in seconds
    frames = np.arange(gradcamp_array.shape[1])
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    
    #y-axis frequency in Hz
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr/2)

    # Ensure directory exists
    output_dir = "Results_outputs"
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.imshow(gradcamp_array, cmap='jet',  origin='lower', aspect='auto', extent=(times[0], times[-1], mel_freqs[0], mel_freqs[-1]))
    #plt.title(f"Gad-CAM for class: {LABEL_DICT[predicted_class]}, Dataset: {dataname}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar()
    plt.show()

    # Save plot
    plt.imshow(gradcamp_array, cmap='jet',  origin='lower', aspect='auto', extent=(times[0], times[-1], mel_freqs[0], mel_freqs[-1]))
    #plt.title(f"Grad-CAM for class: {LABEL_DICT[predicted_class]}, Dataset: {dataname}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar()

    # Construct filename
    filename = f"gradcam_{LABEL_DICT[predicted_class]}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    if LABEL_DICT[predicted_class] == label:
        pred = True
    else:
        pred = False
    _,_,_,_,_, time_steps = saliency_map_processing(upsampled_attr, predicted_class, LABEL_DICT, "GradCam", dataname, data)

    visulize_waveform(data, time_steps, "GradCam", label,dataname)
    return time_steps

#_________________________________________________

def Occlusion_xai(model, data, LABEL_DICT, dataname, label, device):
    '''
       # Occlusion for SER
        Occlusion sensitivity
        (Otsuki, Takumi, et al. "Recognition of Emotional Speech with CNN and Visual Explanations Using Grad-CAM." ICIC express letters. 
        Part B, Applications: an international journal of research and surveys 11.8 (2020): 767-772)'''
    # Put model in eval mode
    model.eval()
    log_mel = get_log_melspe(data)
    log_mel = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0)  
    log_mel = log_mel.to(device).float()
    # Forward input
    output = model(log_mel)
    predicted_class = output.argmax(dim=1).item()

    # Prepare input tensor for Occlusion
    input_tensor = log_mel
    input_tensor.requires_grad = True

    occlusion = Occlusion(model)
    attributions = occlusion.attribute(
    input_tensor,
    strides=(1, 1, 3),              # step size
    sliding_window_shapes=(1, 1, 10),  # occlusion patch
    target=predicted_class,
    )
    # visualize the attributions "occlusion map"
    att_map = attributions.cpu().detach().numpy()
    att_map = np.squeeze(att_map)  # Remove batch and channel dimensions
    # Visualize
    sr = 16000  # Sample rate
    n_mels = att_map.shape[0]  # Number of mel bands
    hop_length = 240  # Hop length used in mel spectrogram generation

    # x-axis time in seconds
    frames = np.arange(att_map.shape[1])
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)


    #y-axis frequency in Hz
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr/2)
    #att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())  # Normalize to [0, 1]
    plt.figure(figsize=(10, 4))
    plt.imshow(att_map, cmap='jet',  origin='lower', aspect='auto', extent=(times[0], times[-1], mel_freqs[0], mel_freqs[-1]))
    #plt.title(f"Occlusion Map for class: {LABEL_DICT[predicted_class]}, Dataset: {dataname}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar()
    plt.show()

    # check if the output directory exists, if not create it
    if not os.path.exists("Results_outputs"):
        print("Creating directory: Results_outputs")
        output_dir = "Results_outputs"
        os.makedirs(output_dir, exist_ok=True)
    else:
        print("Directory already exists: Results_outputs")
        output_dir = "Results_outputs"

    # Save plot
    plt.imshow(att_map, cmap='jet',  origin='lower', aspect='auto', extent=(times[0], times[-1], mel_freqs[0], mel_freqs[-1]))
    #plt.title(f"Occlusion sensitivity for class: {LABEL_DICT[predicted_class]}, Dataset: {dataname}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar()

    # Construct filename
    filename = f"Occlusion sensitivity {LABEL_DICT[predicted_class]} Dataset: {dataname}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    if LABEL_DICT[predicted_class] == label:
        pred = True
    else:
        pred = False

    if isinstance(att_map, np.ndarray):
        att_map = torch.from_numpy(att_map).to(device)

    _,_,_,_,_, time_steps = saliency_map_processing(att_map, predicted_class, LABEL_DICT, "Occlusion sensitivity", dataname, data)

    visulize_waveform(data, time_steps, "Occlusion sensitivity", label,dataname)
    return time_steps


#_________________________________________________________________________________
def saliency_map_processing(upsampled_attr, predicted_class, LABEL_DICT, xai_method, dataname, data):
        ### Saliency map Processing
    '''replace blue cells (low-importance regions) with zeros — apply a threshold to the Grad-CAM 
       values before summing each window. This way, only higher-importance (e.g., "redder") regions contribute.'''
    min_val = upsampled_attr.min().item()
    max_val = upsampled_attr.max().item()
    mean = upsampled_attr.mean().item()

    '''
       a threshold of 15% of the max intensity. This means that any pixel value below this threshold will be set to zero, effectively removing low-importance regions from the visualization.
       You can adjust the 0.15 factor to control how aggressive the thresholding is.
       A lower value will keep more low-importance regions, while a higher value will remove more
       low-importance regions.
       This approach helps to focus on the most salient features in the Grad-CAM visualization, making
       it easier to interpret the results.
       You can also experiment with different thresholds to find the one that works best for your specific use case.
       This is a common practice in visualizing saliency maps or Grad-CAM outputs, as it helps to highlight the most important features while reducing noise from low-importance regions.
       ref: https://discuss.pytorch.org/t/how-to-set-a-threshold-for-grad-cam-visualization/123778/2
           "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    
    '''

    #threshold = min_val + (upsampled_attr.max().item() - min_val) * 0.20  # Adjust threshold as needed
    threshold = mean
    #upsampled_attr[upsampled_attr < threshold] = 0  # Set low-importance regions to zero
    #print(f"Min value (blue): {upsampled_attr.min().item():.4f}")
    #print(f"Max value (red): {upsampled_attr.max().item():.4f}")
    if xai_method == "GradCam":
        saliency_map = upsampled_attr.squeeze(0).cpu().detach()
    elif xai_method == "Occlusion sensitivity":
        saliency_map = upsampled_attr
    elif xai_method == "CRP":
        saliency_map = upsampled_attr.squeeze(0).cpu().detach()
    else:
        raise ValueError("Unsupported XAI method. Use 'GradCam' or 'Occlusion sensitivity'.")

    # parameters
    #from heatmap_proc import sliding_win, overlay_top_windows, get_log_melspe
    win_wid = 10
    top_k = 5
    sr = 16000  # Sample rate
    hop_length = 240  # Hop length used in mel spectrogram generation
    seconds_per_frame = hop_length / sr  # Time per step in seconds

    # Apply sliding window and get top k indices
    top_k_indices, top_windows, ts = sliding_win(saliency_map, win_wid=win_wid, k=top_k, threshold=threshold, xai_method= xai_method)

    #print("Top K Indices:", top_k_indices)
    #print("Top Windows:", top_windows)
    #print("Timestamps:", ts)

    # Save the plot with highlighted top-k windows
    # Ensure directory exists
    output_dir = "Results_outputs"
    if not os.path.exists(output_dir):
        # Create the directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Creating directory: {output_dir}")
    else:
        print(f"Directory already exists: {output_dir}")        

    plt.imshow(saliency_map.squeeze().cpu(), cmap='jet', aspect='auto')
    for (start, end) in ts:
        plt.axvspan(start, end, color='gray', alpha=0.4)
    plt.title(f"{xai_method} highlighted top-k windows for class: {LABEL_DICT[predicted_class]}, {dataname}")
    plt.colorbar()

    # Construct filename
    filename = f"{xai_method}_topk{LABEL_DICT[predicted_class]}-{dataname}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    #_____________ refelect windows on the input spectrogram ______________
    audio_path = data # ds_test.data[index][0]
    audio, sr = librosa.load(audio_path, sr=16000)

    # Generate spectrogram
    spec = get_log_melspe(audio_path, sr)  # shape: [128, T]
    spec_tensor = torch.tensor(spec).unsqueeze(0)  # shape: [1, 128, T]
    
    # Convert time steps (each step ≈ 0.015 sec based on hop_length 240 at 24 kHz)
    time_steps = [(start * seconds_per_frame, end * seconds_per_frame) for start, end in ts]
    print("Top Window Indices:", top_k_indices)
    print("Time steps for projection (s):", time_steps)
    print("\n time steps length is: ",((time_steps[0][1]) - (time_steps[0][0])))

    # Visualization
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spec, sr=sr, hop_length=240, cmap='inferno', y_axis='hz', x_axis='time')
    #plt.title(f"Spectrogram with Highlighted Top Attention Windows of {xai_method}")
    for start, end in ts:
        plt.axvspan(start * seconds_per_frame, end * seconds_per_frame, color='white', alpha=0.4)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
    # Save the spectrogram with highlighted windows
    # Construct filename
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spec, sr=sr, hop_length=240, cmap='inferno', y_axis='mel', x_axis='time')
    plt.title(f"Spectrogram with Highlighted Top Attention Windows of {xai_method}")
    for start, end in ts:
        plt.axvspan(start * seconds_per_frame, end * seconds_per_frame, color='white', alpha=0.4)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    filename = f"{xai_method} topk highlighted on input spectrogram{LABEL_DICT[predicted_class]}-{dataname}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return top_k_indices, top_windows, ts, saliency_map, spec_tensor, time_steps

#_________________________________________________
### Similarity temporal and semantic scores calculation
def explanation_analysis(B_start,B_end, A_start,A_end):
        interval =[(A_start,A_end)]
        Scores = []
        overlap_start = max(B_start,A_start)
        overlap_end = min(B_end,A_end)
        if overlap_start < overlap_end:
            Time_duration = (overlap_end - overlap_start)
            # print("Time duration: ",Time_duration)
            B_duration = B_end - B_start
            # print("B_duration:", B_duration)
            Score = (Time_duration/B_duration)*100
            print("The Aligned time interval is: ", [B_start, B_end])
            # print("Socre:", Score)

        else:
            Score = 0.0

        if Score != 0:
             Scores.append(Score)

        return Scores

#_________________________________________________
 
##________Further visualization
def visulize_waveform(data, time_steps, xai_method, label,dataset):
    audio, sr = librosa.load(data, sr=16000)
    # --- Trim/pad to fixed 2s ---
    target_len = int(sr * 2.0)
    audio = librosa.util.fix_length(audio, size=target_len)
    # Assuming `audio`, `sr`, and `time_steps` are already defined
    plt.figure(figsize=(10, 4))
    time = np.linspace(0, len(audio) / sr, len(audio))

    # Plot the full waveform in gray
    plt.plot(time, audio, label="Waveform", color="gray", alpha=0.6)

    # Define colors for different windows
    colors = ["blue", "green", "red", "orange", "purple"]  # Extend this list as needed
    print(time_steps)
    # Overlay selected windows in different colors
    for i, (start, end) in enumerate(time_steps):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        plt.plot(
            time[(start_sample):(end_sample)],
            audio[(start_sample):(end_sample)],
            color=colors[i % len(colors)],
            label=f"Window {i + 1}",
            linewidth=2,  # Thicker line for better visibility
        )

    # Add labels, legend, and display
    #plt.title(f"{xai_method} Time-Domain Waveform with Highlighted Explanable Windows for {label} on {dataset}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    #plt.legend()
    plt.tight_layout()
    plt.show()
    # save the waveform visualization
    output_dir = "Results_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Creating directory: {output_dir}")
        plt.savefig(os.path.join(output_dir, f"waveform_with_top_windows for{xai_method}on {dataset}.png"), bbox_inches='tight')
    else:
        print(f"Directory already exists: {output_dir}")
    
    Audio(data)























