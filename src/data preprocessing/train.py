import os
import sys
import random
import glob
from pathlib import Path

import pandas as pd
import numpy as np
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# To play audio files
from IPython.display import Audio

# PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import utils as sn

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Successfully imported libraries")

Crema_dir = "data/Crema-D/AudioWAV/"
Ravdess_dir = 'data/audio_speech_actors_01-24/'



