# Standard libraries
import os
import sys
import math
import random
import glob
import csv
from pathlib import Path

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

# IPython utilities for audio
from IPython.display import Audio

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
    samples = librosa.util.fix_length(samples, size=int(sr * 2))  # Ensure length matches 2 seconds

    # Generate mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=samples.astype(np.float32),
        sr=sr,
        n_fft=1024,
        win_length=480,
        hop_length=240,
        n_mels=128
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
    if label == "angry" or label == "fear" or label=="happy":        
            top_loudness_intervals, top_shrillness_intervals,top_jitter,top_shimmer = Top_human_ref(data,label)
            calculate_similarity_scores(heatmap2, top_loudness_intervals, 
                                        top_shrillness_intervals, top_jitter, top_shimmer, time_steps, 
                                        "CRP", dataname, label,pred)
            imgify(heatmap2, symmetric=True)
    elif label == "sad" or label == "neutral":
            lowest_loudness_intervals, lowest_shrillness_intervals,lowest_jitter,lowest_shimmer = Top_human_ref(data, label) 
            calculate_similarity_scores(heatmap2, lowest_loudness_intervals,
                                         lowest_shrillness_intervals, lowest_jitter, lowest_shimmer,time_steps,
                                           "CRP", dataname, label, pred)
            imgify(heatmap2, symmetric=True)
            
     
    else:
        raise ValueError("Unsupported label. Use 'angry', 'fear', 'happy', 'sad', or 'neutral'.")
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
    if label == "angry" or label == "fear" or label=="happy":        
            top_loudness_intervals, top_shrillness_intervals,top_jitter,top_shimmer = Top_human_ref(data,label)
            calculate_similarity_scores(upsampled_attr, top_loudness_intervals, 
                                        top_shrillness_intervals, top_jitter, top_shimmer, time_steps, 
                                        "GradCam", dataname, label,pred)
    elif label == "sad" or label == "neutral":
            lowest_loudness_intervals, lowest_shrillness_intervals,lowest_jitter,lowest_shimmer = Top_human_ref(data, label) 
            calculate_similarity_scores(upsampled_attr, lowest_loudness_intervals,
                                         lowest_shrillness_intervals, lowest_jitter, lowest_shimmer,time_steps,
                                           "GradCam", dataname, label, pred)
            
     
    else:
        print(label)
        raise ValueError("Unsupported label. Use 'angry', 'fear', 'happy', 'sad', or 'neutral'.")
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
    plt.imshow(
    att_map,
    cmap='jet',
    origin='lower',
    aspect='auto',
    extent=(times[0], times[-1], mel_freqs[0] / 1000, mel_freqs[-1] / 1000)  # convert to kHz
     )
    # set y-ticks in kHz
    max_khz = mel_freqs[-1] / 1000
    yticks = np.arange(0, np.ceil(max_khz) + 1e-9, 2.0)
    plt.yticks(yticks)
    #plt.imshow(att_map, cmap='jet',  origin='lower', aspect='auto', extent=(times[0], times[-1], mel_freqs[0], mel_freqs[-1]))
    #plt.title(f"Occlusion Map for class: {LABEL_DICT[predicted_class]}, Dataset: {dataname}")
    #plt.xlabel("Time (s)")
    #plt.ylabel("Frequency (Hz)")
    plt.colorbar()
    plt.tight_layout()
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
    #segment_and_visualize(data, upsampled_attr)
    if label == "angry" or label == "fear" or label=="happy":     
            top_loudness_intervals, top_shrillness_intervals,top_jitter, top_shimmer = Top_human_ref(data,label)            
            calculate_similarity_scores(att_map, top_loudness_intervals, 
                                        top_shrillness_intervals, top_jitter, top_shimmer, time_steps, 
                                        "Occlusion sensitivity", dataname, label,pred)
    elif label == "sad" or label == "neutral":
            lowest_loudness_intervals, lowest_shrillness_intervals,lowest_jitter, lowest_shimmer= Top_human_ref(data, label) 
            calculate_similarity_scores(att_map, lowest_loudness_intervals,
                                         lowest_shrillness_intervals, lowest_jitter, lowest_shimmer,time_steps,
                                           "Occlusion sensitivity", dataname, label,pred)
    else:
        raise ValueError("Unsupported label. Use 'angry', 'fear', 'happy', 'sad', or 'neutral'.")
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

'''
#### Human-referenced vocal cues processing
Extract the salient windows from the audio file
Extract features of interests (Pyhsycoacoustics)
- High-Frequency Energy (HF 500) ...Shrillness ( When the pitch of the sound is higher, the sound is shrill)
- Voice Intensity-->Loudness
- Mean (F0)-->	Average Pitch
- SD (F0)	-->Pitch Range
- Speech Rate	-->Speaking Rate
- Pause Proportion	-->Proportion of Pauses

Analyze Shrillness and loudness for class Angry: instance level (e.g instance: 'data/Crema-D/AudioWAV/1058_WSI_ANG_XX.wav' from the test data)

First layer
conditions = [{"y": [label], "layer1.conv1": [58]}]
'''
'''def Top_human_ref(data, label):
    """    Extracts and analyzes human-referenced vocal cues from audio data.
    This function processes a specific audio sample from the test dataset to compute vocal cues features.
    It calculates the top-k time intervals where these features exceed their average values.

    Applies pydub preprocessing: trims the first non-silent interval,
    converts to mono, resamples to 24 kHz, converts to NumPy, and
    fixes length to 2 seconds before feature extraction.
    """
    # Load audio file
    audio_path = data #ds_test.data[index][0]
    y, sr = librosa.load(audio_path, sr=16000)
    print(sr)
    
    

    # Window parameters
    window_size = int(sr * 0.15)  # 0.15 second window
    hop_length = int(sr * 0.25)  # 0.25 second hop
    overlap_length = window_size - hop_length  # Calculate overlap length
    print(f"Window Size: {window_size}, Hop Length: {hop_length}, Overlap Length: {overlap_length}")


    # Calculate loudness (RMS)
    loudness = librosa.feature.rms(y=y, frame_length=window_size, hop_length=hop_length).flatten()

    # Calculate shrillness (Spectral Centroid)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length).flatten()

    # Convert to Parselmouth Sound object for jitter/shimmer
    sound = parselmouth.Sound(audio_path)
    # Extract jitter (%)
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

    # Extract shimmer (%)
    shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    # Approximate frame-wise jitter and shimmer by splitting signal into windows
    framewise_jitter = []
    framewise_shimmer = []

    
    for i in range(0, len(y) - window_size, hop_length):
        frame = y[i:i+window_size]
        temp_sound = parselmouth.Sound(frame, sampling_frequency=sr)
        temp_pp = call(temp_sound, "To PointProcess (periodic, cc)", 75, 500)
        try:
            frame_jitter = call(temp_pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            frame_shimmer = call([temp_sound, temp_pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except:
            frame_jitter, frame_shimmer = 0, 0  # fallback in case of error
        framewise_jitter.append(frame_jitter)
        framewise_shimmer.append(frame_shimmer)

    framewise_jitter = np.array(framewise_jitter)
    framewise_shimmer = np.array(framewise_shimmer) 

    # Convert indices to time intervals
    def time_step(idx, hop_length, sr):
        start = (idx * hop_length) / sr
        end = ((idx * hop_length) + window_size) / sr
        return (start, end)
    
    # High-arousal emotions
    if label == "angry" or label == "fear" or label=="happy":
        # Find top-k time intervals
        def top_k_intervals(values, k=5):
            values = np.asarray(values)
            mean_value = np.mean(values)
            # Select values below the mean
            above_avg_indices = np.where(values < mean_value)[0]
            above_avg_values = values[above_avg_indices]
            top_k_indices = above_avg_indices[np.argsort(above_avg_values)[-k:][::-1]]
            top_k_indices = np.argsort(values)[-k:][::-1]  # Indices of top-k values in descending order
            return [time_step(idx, hop_length, sr) for idx in top_k_indices]

        # Top 5 time intervals
        top_loudness_intervals = top_k_intervals(loudness, k=5)
        top_shrillness_intervals = top_k_intervals(spectral_centroid, k=5)
        top_jitter_intervals = top_k_intervals(framewise_jitter, k=5)
        top_shimmer_intervals = top_k_intervals(framewise_shimmer, k=5)

        # Print results
        print("Top 5 Time Intervals of Loudness above avg value:", top_loudness_intervals)
        print("Top 5 Time Intervals of Shrillness above avg value:", top_shrillness_intervals)
        print("Top 5 Time Intervals of Jitter Intervals:", top_jitter_intervals)
        print("Top 5 Time Intervals of Shimmer Intervals:", top_shimmer_intervals)
        
        # Calculate the total number of intervals for each feature
        total_loudness_intervals = len(loudness)
        total_shrillness_intervals = len(spectral_centroid)
        total_jitter_intervals = len(framewise_jitter)
        total_shimmer_intervals = len(framewise_shimmer)

        # Print the total number of intervals
        print("Total human-referenced segments for Loudness:", total_loudness_intervals)
        print("Total human-referenced segments for Shrillness:", total_shrillness_intervals)
        print("Total human-referenced segments for Jitter:", total_jitter_intervals)
        print("Total human-referenced segments for Shimmer:", total_shimmer_intervals)
        return top_loudness_intervals, top_shrillness_intervals, top_jitter_intervals, top_shimmer_intervals
    
    # Low-arousal emotions
    elif label == "sad" or label == "neutral":
        def bottom_k_intervals(values, k):
                values = np.asarray(values)
                bottom_k_indices = np.argsort(values)[:k]  # Indices of bottom-k values in  order
                return [time_step(idx, hop_length, sr) for idx in bottom_k_indices]
        # Top 5 time intervals
        #top_loudness_intervals = top_k_intervals(loudness, k=5)
        #top_shrillness_intervals = top_k_intervals(spectral_centroid, k=5)
        lowest_loudness = bottom_k_intervals(loudness, k = 5)
        lowest_shrillness = bottom_k_intervals(spectral_centroid, k =5)
        lowest_jitter = bottom_k_intervals(framewise_jitter, k=5)
        lowest_shimmer = bottom_k_intervals(framewise_shimmer, k=5)

        # Print results
        print("lowest_loudness:  ", lowest_loudness)
        print("lowest shrillness: ", lowest_shrillness)
        print("Lowest Jitter:", lowest_jitter)
        print("Lowest Shimmer:", lowest_shimmer)
        return lowest_loudness, lowest_shrillness, lowest_jitter, lowest_shimmer
    else:
        raise ValueError("Unsupported label. Use 'angry', 'fear', 'happy', 'sad', or 'neutral'.")
'''

def Top_human_ref(data, label):
    """
    Extract vocal-cue intervals after preprocessing:
      - trim to first non-silent region,
      - convert to mono,
      - resample to 16 kHz,
      - fix length to 2.0 s,
      - window = 30 ms, hop = 15 ms,
    then return top/bottom intervals for loudness (RMS), spectral centroid,
    and frame-wise jitter/shimmer (Parselmouth).
    """
    # Load arbitrary audio file/bytes path with pydub
    audio = AudioSegment.from_file(data)

    # Detect non-silent regions (pydub works in milliseconds)
    '''The term dB FS (or dBFS) means decibels relative to full scale. It is used for amplitude levels in digital systems with a maximum available peak level, e.g., PCM encoding, where 0 dB FS is assigned to the maximum level. A signal that reaches 50 percent of the maximum level would, for example, have a value of -6 dB FS. All peak measurements will be negative numbers.'''
    silence_thresh = audio.dBFS - 14  # ~14 dB below average loudness
    '''Recommended min_silence_len values (speech emotion datasets, 16 kHz audio):

        200–300 ms → good default. Keeps micro-pauses and hesitation that are emotionally relevant, while trimming clear gaps.

        ≤150 ms → too aggressive; may cut natural pauses in emotional speech.

        ≥500 ms → safer for noisy data, but risks leaving long silences at beginning/end'''
    nonsilent = detect_nonsilent(audio, min_silence_len=350, silence_thresh=silence_thresh)

    # Keep the first non-silent interval if present
    if nonsilent:
        start_ms, end_ms = nonsilent[0][0], nonsilent[-1][1]
        audio = audio[start_ms:end_ms]

    # Ensure mono and target sampling rate
    sr = 16000 
    audio = audio.set_channels(1).set_frame_rate(sr)

    # Convert to NumPy (int16 -> float32 in [-1, 1])
    samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
    y = np.array(audio.get_array_of_samples(), dtype=np.int16).astype(np.float32) / 32768.0

    # Ensure a fixed length of 2 seconds (pad/truncate at the end)
    target_len = int(sr * 2.0)
    y = librosa.util.fix_length(y, size=target_len)

    # Windowing params (kept as in your code) 
    '''From Data processing section: a window length of 480 samples, and a hop size of 240 samples
    Then:
      Window duration = 480 / 16000 = 0.030 seconds (20 ms)
      Hop duration = 240 / 16000 = 0.015 seconds (10 ms)
      Overlap = 480 - 240 = 240 samples (i.e., 50% overlap).
    '''
    window_size = int(sr * 0.10)   # 0.15 s window
    hop_length = int(sr * 0.03)    # 0.25 s hop
    overlap_length = window_size - hop_length
    print(f"Window Size: {window_size}, Hop Length: {hop_length}, Overlap Length: {overlap_length}")

    # Ensure y is at least window_size long
    if len(y) < window_size:
        raise ValueError(f"Audio length {len(y)} is shorter than the window size {window_size}.")
    
    '''TODO: Apply audio segmentation before feature extraction'''

    #_____________________________________
    # --- Features ---
    loudness = librosa.feature.rms(
        y=y, frame_length = window_size, hop_length=hop_length, center=False
    ).flatten()
    # Calculate spectral centroid (shrillness)
    # Spectral centroid is a measure of the "center of mass" of the spectrum, often associated with shrillness
    # It indicates how "high" the sound is perceived
    # Higher values indicate a more shrill sound, while lower values indicate a deeper sound.
    # It is calculated as the weighted mean of the frequencies present in the sound, weighted by their magnitudes.
    # It is often used in audio analysis to characterize the timbre of a sound.
    # librosa.feature.spectral_centroid returns the spectral centroid in Hz
    # The result is a 2D array, so we flatten it to get a 1D array.
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=window_size,win_length=window_size, hop_length=hop_length, center=False
    ).flatten()
    #spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length).flatten()


    # Parselmouth (Praat) jitter/shimmer on the preprocessed waveform
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    point_process = call(sound, "To PointProcess (periodic, cc)", 90, 500)
    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    # Approximate frame-wise jitter/shimmer
    framewise_jitter, framewise_shimmer = [], []
    for i in range(0, len(y) - window_size + 1, hop_length):
        frame = y[i:i + window_size]
        temp_sound = parselmouth.Sound(frame, sampling_frequency=sr)
        temp_pp = call(temp_sound, "To PointProcess (periodic, cc)", 90, 500)
        try:
            fj = call(temp_pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            fs = call([temp_sound, temp_pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except Exception:
            fj, fs = 0.0, 0.0
        framewise_jitter.append(fj)
        framewise_shimmer.append(fs)

    framewise_jitter = np.asarray(framewise_jitter)
    framewise_shimmer = np.asarray(framewise_shimmer)

    # Index -> (start, end) in seconds
    def time_step(idx, hop_length, sr):
        start = (idx * hop_length) / sr
        end = ((idx * hop_length) + window_size) / sr
        return (start, end)
    
    # returns the indices of the biggest k values, in decending order (higher -> lower).
    def top_k_intervals(values, k=5):
            values = np.asarray(values)
            top_k_indices = np.argsort(values)[-k:][::-1]
            return [time_step(idx, hop_length, sr) for idx in top_k_indices]
    
    # returns the indices of the smallest k values, in ascending order (lowest -> higher).
    def bottom_k_intervals(values, k):
            values = np.asarray(values)
            idx = np.argsort(values)[:k]
            return [time_step(i, hop_length, sr) for i in idx]
    # High-arousal emotions
    # angry, fear, happy
    if label in {"angry", "fear", "happy"}:
        top_loudness_intervals = top_k_intervals(loudness, k=5)
        top_shrillness_intervals = top_k_intervals(spectral_centroid, k=5)
        top_jitter_intervals = top_k_intervals(framewise_jitter, k=5)
        top_shimmer_intervals = top_k_intervals(framewise_shimmer, k=5)

        print("Top 5 Time Intervals of Loudness above avg value:", top_loudness_intervals)
        print("Top 5 Time Intervals of Shrillness above avg value:", top_shrillness_intervals)
        print("Top 5 Time Intervals of Jitter Intervals:", top_jitter_intervals)
        print("Top 5 Time Intervals of Shimmer Intervals:", top_shimmer_intervals)

        print("Total human-referenced segments for Loudness:", len(loudness))
        print("Total human-referenced segments for Shrillness:", len(spectral_centroid))
        print("Total human-referenced segments for Jitter:", len(framewise_jitter))
        print("Total human-referenced segments for Shimmer:", len(framewise_shimmer))

        return top_loudness_intervals, top_shrillness_intervals, top_jitter_intervals, top_shimmer_intervals
    # Low-arousal emotions
    # sad, neutral
    elif label in {"sad", "neutral"}:
        lowest_loudness = bottom_k_intervals(loudness, k=5)
        lowest_shrillness = bottom_k_intervals(spectral_centroid, k=5)
        lowest_jitter = bottom_k_intervals(framewise_jitter, k=5)
        lowest_shimmer = bottom_k_intervals(framewise_shimmer, k=5)

        print("lowest_loudness:", lowest_loudness)
        print("lowest_shrillness:", lowest_shrillness)
        print("Lowest Jitter:", lowest_jitter)
        print("Lowest Shimmer:", lowest_shimmer)

        return lowest_loudness, lowest_shrillness, lowest_jitter, lowest_shimmer

    else:
        raise ValueError("Unsupported label. Use 'angry', 'fear', 'happy', 'sad', or 'neutral'.")



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

#___________________________________________Similarity Scores Calculation_________________________
def calculate_similarity_scores(upsampled_attr, loudness_intervals, shrillness_intervals,  jitter_intervals, shimmer_intervals,
                                 time_steps, xai_method, dataset, Emotion, pred):
    
    k = 5  # Number of top intervals to consider
    output_file = "explanation_TS_Eval.csv"


    print("_____________________Loudness__________________________")
    Loudness_Scores = []
    for B_start, B_end in time_steps:
        for A_start,A_end in loudness_intervals:
            #print(B_start, B_end, A_start,A_end)
            score = explanation_analysis(B_start, B_end, A_start,A_end)
            Loudness_Scores.append(score)
    data = [item for sublist in Loudness_Scores if sublist for item in sublist]
    print(data)
    print("Score: ",data)
    if len(data) == 0:
        print(f"No overlapping intervals found between Saliency map and human-referenced segments for {xai_method} on {dataset}.")
        pass
    else:
        print(f"Final Explanation Score of Loudness for {xai_method} on {dataset} is: " ,sum(data)/len(data))
        print(f"Final Explanation Score coverage rate of Loudness for {xai_method} on {dataset} is: " ,sum(data)/k)
        print(f"How many intervals that exp windows belongs to in the GT windows for {xai_method} on {dataset}: {len(data)}\n")

    print("_____________________Shrillness__________________________")

    shrillness_Scores = []
    for B_start, B_end in time_steps:
        for A_start,A_end in shrillness_intervals:
            #print(B_start, B_end, A_start,A_end)
            score = explanation_analysis(B_start, B_end, A_start,A_end)
            shrillness_Scores.append(score)
    data2 = [item for sublist in shrillness_Scores if sublist for item in sublist]

    print("Score: ",data2)
    if len(data2) == 0:
        print(f"No overlapping intervals found between Saliency map and human-referenced segments for {xai_method} on {dataset}")
        pass
    else:
        print(f"Final Explanation Score of Shrillness for {xai_method} on {dataset} is: " ,sum(data2)/len(data2))
        print(f"Final Explanation Score coverage rate of Shrillness for {xai_method} on {dataset} is: " ,sum(data2)/k)
        print(f"How many intervals that exp windows belongs to in the GT windows for {xai_method} on {dataset}: {len(data2)}\n")

    print("_______________________Jitter__________________________")
    jitter_Scores = []
    for B_start, B_end in time_steps:
        for A_start,A_end in jitter_intervals:
            #print(B_start, B_end, A_start,A_end)
            score = explanation_analysis(B_start, B_end, A_start,A_end)
            jitter_Scores.append(score)
    data3 = [item for sublist in jitter_Scores if sublist for item in sublist]

    print("Score: ",data3)
    if len(data3) == 0:
        print(f"No overlapping intervals found between Saliency map and human-referenced segments for {xai_method} on {dataset}")
        pass
    else:
        print(f"Final Explanation Score of Jitter for {xai_method} on {dataset} is: " ,sum(data3)/len(data3))
        print(f"Final Explanation Score coverage rate of Jitter for {xai_method} on {dataset} is: " ,sum(data3)/k)
        print(f"How many intervals that exp windows belongs to in the GT windows for {xai_method} on {dataset}: {len(data3)}\n")
    print("_______________________Shimmer__________________________")
    shimmer_Scores = []
    for B_start, B_end in time_steps:
        for A_start,A_end in shimmer_intervals:
            #print(B_start, B_end, A_start,A_end)
            score = explanation_analysis(B_start, B_end, A_start,A_end)
            shimmer_Scores.append(score)
    data4 = [item for sublist in shimmer_Scores if sublist for item in sublist]     
    print("Score: ",data4)
    if len(data4) == 0: 
        print(f"No overlapping intervals found between Saliency map and human-referenced segments for {xai_method} on {dataset}")
        pass
    else:   
        print(f"Final Explanation Score of Shimmer for {xai_method} on {dataset} is: " ,sum(data4)/len(data4))
        print(f"Final Explanation Score coverage rate of Shimmer for {xai_method} on {dataset} is: " ,sum(data4)/k)
        print(f"How many intervals that exp windows belongs to in the GT windows for {xai_method} on {dataset}: {len(data4)}\n")

    # Save results to CSV
    # Check if file exists and determine round
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            round_numbers = [int(row[0]) for row in rows[1:]] if len(rows) > 1 else []
            round_number = max(round_numbers) + 1 if round_numbers else 1
    else:
        round_number = 1


    # Prepare rows
    results = []

    if len(data) == 0:
        loudness_score = coverage_rate_loudness = interval_count_loudness = None
    else:
        loudness_score = sum(data) / len(data)
        coverage_rate_loudness = sum(data) / k
        interval_count_loudness = len(data)

    results.append([
        round_number,        
        xai_method,
        dataset,
        Emotion,
        "Loudness",
        loudness_score,
        coverage_rate_loudness,
        interval_count_loudness, 
        pred
    ])

    if len(data2) == 0:
        shrillness_score = coverage_rate_shrillness = interval_count_shrillness = None
    else:
        shrillness_score = sum(data2) / len(data2)
        coverage_rate_shrillness = sum(data2) / k
        interval_count_shrillness = len(data2)

    results.append([
        round_number,        
        xai_method,
        dataset,
        Emotion,
        "Shrillness",
        shrillness_score,
        coverage_rate_shrillness,
        interval_count_shrillness,
        pred
    ])
    if len(data3) == 0:
        jitter_score = coverage_rate_jitter = interval_count_jitter = None
    else:
        jitter_score = sum(data3) / len(data3)
        coverage_rate_jitter = sum(data3) / k
        interval_count_jitter = len(data3)

    results.append([
        round_number,        
        xai_method,
        dataset,
        Emotion,
        "Jitter",
        jitter_score,
        coverage_rate_jitter,
        interval_count_jitter,
        pred
    ])

    if len(data4) == 0:
        shimmer_score = coverage_rate_shimmer = interval_count_shimmer = None
    else:
        shimmer_score = sum(data4) / len(data4)
        coverage_rate_shimmer = sum(data4) / k
        interval_count_shimmer = len(data4)
    results.append([
        round_number,        
        xai_method,
        dataset,
        Emotion,
        "Shimmer",
        shimmer_score,
        coverage_rate_shimmer,
        interval_count_shimmer,
        pred
    ])

    # Write or append to CSV
    file_exists = os.path.isfile(output_file)

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow([
                "Round", "XAI Method", "Dataset", "Emotion",
                "Vocal Cue", "Similarity Score", "Mean Coverage Rate",
                  "Similar Interval Count", "pred"
            ])
        writer.writerows(results)
        print(f"Results saved to {output_file}")


 
##________Further visualization
def visulize_waveform(data, time_steps, xai_method, label,dataset):
    audio, sr = librosa.load(data, sr=16000)
    # Assuming `audio`, `sr`, and `time_steps` are already defined
    # --- Trim/pad to fixed 2s ---
    target_len = int(sr * 2.0)
    audio = librosa.util.fix_length(audio, size=target_len)

    # --- Build time axis ---
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
















#__________________
def calculate_similarity_scores_1(upsampled_attr, loudness_intervals, shrillness_intervals, jitter_intervals, shimmer_intervals,
                                time_steps, xai_method, dataset, Emotion, pred):
    k = 5  # Number of top intervals to consider
    output_file = "explanation_TS_Eval.csv"
    
    attributes = {
        "Loudness": loudness_intervals,
        "Shrillness": shrillness_intervals,
        "Jitter": jitter_intervals,
        "Shimmer": shimmer_intervals
    }
    
    results = []
    
    def compute_scores(attr_name, intervals):
        print(f"_____________________{attr_name}__________________________")
        scores = [
            score
            for B_start, B_end in time_steps
            for A_start, A_end in intervals
            for score in (explanation_analysis(B_start, B_end, A_start, A_end),)
        ]
        flat_scores = [s for sub in scores if sub for s in sub]
        print("Score: ", flat_scores)

        if not flat_scores:
            print(f"No overlapping intervals found for {xai_method} on {dataset}.")
            return "N/A", "N/A", "N/A"
        
        mean_score = sum(flat_scores) / len(flat_scores)
        coverage_rate = sum(flat_scores) / k
        interval_count = len(flat_scores)
        
        print(f"Final Explanation Score of {attr_name}: {mean_score}")
        print(f"Coverage Rate of {attr_name}: {coverage_rate}")
        print(f"Interval Count of {attr_name}: {interval_count}\n")
        
        return mean_score, coverage_rate, interval_count
    
    # Determine round number
    round_number = 1
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) > 1:
                round_numbers = [int(row[0]) for row in rows[1:]]
                round_number = max(round_numbers) + 1

    # Compute scores and collect results
    for attr_name, intervals in attributes.items():
        score, coverage, count = compute_scores(attr_name, intervals)
        results.append([
            round_number, xai_method, dataset, Emotion,
            attr_name, score, coverage, count, pred
        ])
    
    # Save to CSV
    write_header = not os.path.isfile(output_file)
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "Round", "XAI Method", "Dataset", "Emotion",
                "Vocal Cue", "Similarity Score", "Mean Coverage Rate",
                "Similar Interval Count", "pred"
            ])
        writer.writerows(results)
    print(f"Results saved to {output_file}")



























#__________________________________________dataset: 1_________________________
def CremaD_processing(data_path, data_setname, sample_rate=16000):
    """    Process the Cream-D dataset to extract audio features and labels.
    Args:
        data_path (str): Path to the Cream-D dataset.
        data_setname (str): Name of the dataset.
        sample_rate (int): Sample rate for audio processing."
    Returns:
        pd.DataFrame: DataFrame containing audio features and labels.
    """
    LABEL_DICT = {0:'fear', 1:'neutral', 2:'happy',3:'angry', 4:'disgust', 5:'surprise',6:'sad'}
    LABELS = list(LABEL_DICT.values())
    data_path = data_path
    Crema_df = pd.read_pickle('Crema_df.pkl')
     
    # Split the DataFrame into training and testing sets
    X= Crema_df.iloc[:, 1:].values
    Y = Crema_df['Emotions'].values
    x_train_v, x_test, y_train_v, y_test = train_test_split(X, Y, random_state=42,test_size=0.1, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(X, Y, random_state=42,test_size=0.1, shuffle=True)
    x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape

    # Dataloader
    class getdata(Dataset):
        def __init__(self, data, labels, transform=None):
            """
            Args:
                data (list of tuples): Each element is a tuple (file_path, spectrogram_array).
                labels (list): List of integer labels corresponding to the data.
            """
            self.data = data
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # Retrieve the spectrogram and label][[[]]]]]                
            _, spectrogram = self.data[idx]
            label = self.labels[idx]

            # Convert to torch tensor and add channel dimension
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
            label = torch.tensor(label, dtype=torch.long)

            # Apply any transformation
            if self.transform:
                spectrogram = self.transform(spectrogram)

            return spectrogram, label
    ds_train = getdata(x_train, y_train)
    ds_test = getdata(x_test, y_test)
    ds_val = getdata(x_val, y_val)

    dl_train = DataLoader(ds_train, batch_size=2, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=2, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=2, shuffle=True)

    return ds_train, ds_test, ds_val, dl_train, dl_test, dl_val, LABELS

#_________________________________dataset: 2_________________________
def SAVEE_processing(data_path, data_setname, sample_rate=16000):
    savee_df = pd.read_pickle('SAVEE_df.pkl')
    X= savee_df.iloc[:, 1:].values
    Y = savee_df['Emotions'].values
    x_train_v, x_test, y_train_v, y_test = train_test_split(X, Y, random_state=42,test_size=0.1, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(X, Y, random_state=42,test_size=0.1, shuffle=True)
    x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape
    # Dataloader
    class getdata(Dataset):
        def __init__(self, data, labels, transform=None):
            """
            Args:
                data (list of tuples): Each element is a tuple (file_path, spectrogram_array).
                labels (list): List of integer labels corresponding to the data.
            """
            self.data = data
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # Retrieve the spectrogram and label][[[]]]]]                
            _, spectrogram = self.data[idx]
            label = self.labels[idx]

            # Convert to torch tensor and add channel dimension
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
            label = torch.tensor(label, dtype=torch.long)

            # Apply any transformation
            if self.transform:
                spectrogram = self.transform(spectrogram)

            return spectrogram, label
    ds_train = getdata(x_train, y_train)
    ds_test = getdata(x_test, y_test)
    ds_val = getdata(x_val, y_val)

    dl_train = DataLoader(ds_train, batch_size=2, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=2, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=2, shuffle=True)
    return ds_train, ds_test, ds_val, dl_train, dl_test, dl_val, savee_df['Emotions'].unique().tolist()

#______________________________dataset: 3_________________________

def RAVDESS_processing(data_path, data_setname, sample_rate=16000):
    rav_df = pd.read_pickle('Ravdess.pkl')
    X= rav_df.iloc[:, 1:].values
    Y = rav_df['emotion'].values
    x_train_v, x_test, y_train_v, y_test = train_test_split(X, Y, random_state=42,test_size=0.1, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(X, Y, random_state=42,test_size=0.1, shuffle=True)
    x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape
    # Dataloader
    class getdata(Dataset):
        def __init__(self, data, labels, transform=None):
            """
            Args:
                data (list of tuples): Each element is a tuple (file_path, spectrogram_array).
                labels (list): List of integer labels corresponding to the data.
            """
            self.data = data
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # Retrieve the spectrogram and label][[[]]]]]                
            _, spectrogram = self.data[idx]
            label = self.labels[idx]

            # Convert to torch tensor and add channel dimension
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
            label = torch.tensor(label, dtype=torch.long)

            # Apply any transformation
            if self.transform:
                spectrogram = self.transform(spectrogram)

            return spectrogram, label
    ds_train = getdata(x_train, y_train)
    ds_test = getdata(x_test, y_test)
    ds_val = getdata(x_val, y_val)

    dl_train = DataLoader(ds_train, batch_size=2, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=2, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=2, shuffle=True)
    return ds_train, ds_test, ds_val, dl_train, dl_test, dl_val, rav_df['emotion'].unique().tolist()

#_______________________________dataset: 4_________________________
def TESS_processing(data_path, data_setname, sample_rate=16000):
    Tess_df = pd.read_pickle('TESS_df.pkl')
    X= Tess_df.iloc[:, 1:].values
    Y = Tess_df['Emotions'].values
    x_train_v, x_test, y_train_v, y_test = train_test_split(X, Y, random_state=42,test_size=0.1, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(X, Y, random_state=42,test_size=0.1, shuffle=True)
    x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape
    # Dataloader
    class getdata(Dataset):
        def __init__(self, data, labels, transform=None):
            """
            Args:
                data (list of tuples): Each element is a tuple (file_path, spectrogram_array).
                labels (list): List of integer labels corresponding to the data.
            """
            self.data = data
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # Retrieve the spectrogram and label][[[]]]]]                
            _, spectrogram = self.data[idx]
            label = self.labels[idx]

            # Convert to torch tensor and add channel dimension
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
            label = torch.tensor(label, dtype=torch.long)

            # Apply any transformation
            if self.transform:
                spectrogram = self.transform(spectrogram)

            return spectrogram, label
    ds_train = getdata(x_train, y_train)
    ds_test = getdata(x_test, y_test)
    ds_val = getdata(x_val, y_val)

    dl_train = DataLoader(ds_train, batch_size=2, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=2, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=2, shuffle=True)
    return ds_train, ds_test, ds_val, dl_train, dl_test, dl_val, Tess_df['Emotions'].unique().tolist()

#