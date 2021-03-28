import os
from multiprocessing import cpu_count

# Project files
from src.dataset import TrainDataset
from src.learner import Learner
from src.loss_functions import costFunction as lossFunction

from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop,
    HorizontalFlip, VerticalFlip, RandomBrightness, RandomContrast,
    RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose, Blur
)
from albumentations.pytorch import ToTensorV2


# Data paths
DATA_ROOT = os.path.realpath("../bms-molecular-translation-data")
TRAIN_X_DIR = os.path.join(DATA_ROOT, "train")
TRAIN_Y_FIL = os.path.join(DATA_ROOT, "train_labels.csv")
VALID_X_DIR = TRAIN_X_DIR
VALID_Y_FIL = TRAIN_Y_FIL

# Model parameters
LOAD_MODEL = False
MODEL_PATH = None
BATCH_SIZE = 128
PATCH_SIZE = 256
PATIENCE = 10
LEARNING_RATES = [5e-4, 1e-3]
DROP_LAST_BATCH = False
NUMBER_OF_DATALOADER_WORKERS = cpu_count()
NUMBER_OF_FOLDS = 5


def getTrainFilePath(image_id):
    return "{}/{}/{}/{}/{}.png".format(
        TRAIN_X_DIR, image_id[0], image_id[1], image_id[2], image_id)


def main():
    import pandas as pd
    import torch
    from sklearn.model_selection import StratifiedKFold
    train = pd.read_pickle("train.pkl")
    train["file_path"] = train["image_id"].apply(getTrainFilePath)
    print(train["file_path"])
    tokenizer = torch.load("tokenizer.pth")
    print(f"tokenizer.stoi: {tokenizer.stoi}")
    print(train.keys())
    print(train["file_path"][0])
    folds = train.copy()
    fold = StratifiedKFold(
        n_splits=NUMBER_OF_FOLDS, shuffle=True, random_state=42)
    for n, (train_index, val_index) in enumerate(
            fold.split(folds, folds["InChI_length"])):
        folds.loc[val_index, "fold"] = int(n)
    folds["fold"] = folds["fold"].astype(int)
    print(folds.groupby(["fold"]).size())
    train_transforms = Compose([
        Resize(224, 224),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    valid_transforms = Compose([
        Resize(224, 224),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds['InChI'].values
    train_dataset = TrainDataset(train_folds, tokenizer, train_transforms)
    valid_dataset = TrainDataset(valid_folds, tokenizer, valid_transforms)
    learner = Learner(
        train_dataset, valid_dataset, BATCH_SIZE, LEARNING_RATES, lossFunction,
        tokenizer, PATIENCE, NUMBER_OF_DATALOADER_WORKERS, LOAD_MODEL,
        MODEL_PATH, DROP_LAST_BATCH)
    learner.train()


main()
