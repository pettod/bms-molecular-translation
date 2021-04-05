import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class TrainDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.file_paths = df["file_path"].values
        self.labels = df["InChI_text"].values
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        label = self.labels[idx]
        label = torch.LongTensor(self.tokenizer.text_to_sequence(label))
        label_length = len(label)
        label_length = torch.LongTensor([label_length])
        return image, label, label_length


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.file_paths = df["file_path"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image
