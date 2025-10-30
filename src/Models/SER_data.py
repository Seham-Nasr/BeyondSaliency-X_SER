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
import librosa.display

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
import argparse
import sys
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split


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
            # Retrieve the spectrogram and label             
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


def main():
    parser = argparse.ArgumentParser(description="Run TESS data preparation.")
    parser.add_argument("--data_path", required=True, help="Path to the TESS_df.pkl file")
    parser.add_argument("--dataset_name", required=True, help="Dataset name (for labeling or saving)")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    args = parser.parse_args()

    print(f"[INFO] Loading dataset '{args.dataset_name}' from {args.data_path}")
    outputs = TESS_processing(args.data_path, args.dataset_name, args.sample_rate)

    ds_train, ds_test, ds_val, dl_train, dl_test, dl_val, classes = outputs
    print(f"[INFO] Classes detected: {classes}")
    print(f"[INFO] Train batches: {len(dl_train)}, Val: {len(dl_val)}, Test: {len(dl_test)}")


# python Models/SER_data.py --data_path ./TESS_df.pkl --dataset_name TESS --sample_rate 16000
if __name__ == "__main__":
    sys.exit(main())