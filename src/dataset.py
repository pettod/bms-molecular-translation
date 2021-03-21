import cv2
from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, input_path, target_file_path, transform=None):
        self.input_image_paths = np.array(sorted(glob(f"{input_path}/*/*/*/*.png")))
        self.target_labels = pd.read_csv(target_file_path)
        self.transform = transform

    def __len__(self):
        return len(self.input_image_paths)

    def __getitem__(self, sample_index):
        input_image = Image.open(self.input_image_paths[sample_index])
        target_labels = self.target_labels.iloc[sample_index]
        if self.transform:
            input_image = self.transform(input_image)
        return input_image, target_image
