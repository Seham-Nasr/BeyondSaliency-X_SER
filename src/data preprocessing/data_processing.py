import os
import sys
import pandas as pd
import numpy as np
import librosa
import librosa.display
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from pathlib import Path
import seaborn as sns

# To play audio files
import IPython.display as ipd
from IPython.display import Audio

# PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from pathlib import Path
import random
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def main():
    if len(sys.argv) != 3:
        print("Usage: python dataprocessing.py <data_path> <data_name>")
        sys.exit(1)

    data_path = sys.argv[1]
    data_name = sys.argv[2]

    print(f"Data path: {data_path}")
    print(f"Data name: {data_name}")

if __name__ == "__main__":
    main()

