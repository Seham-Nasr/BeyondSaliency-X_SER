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
