from torchvision import transforms
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop,
    HorizontalFlip, VerticalFlip, RandomBrightness, RandomContrast,
    RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose, Blur
)
from albumentations.pytorch import ToTensorV2

# Project files
from config import CONFIG
from src.dataset import TrainDataset as Dataset
from src.learner import Learner


def getTrainFilePath(image_id):
    return "{}/{}/{}/{}/{}.png".format(
        CONFIG.TRAIN_X_DIR, image_id[0], image_id[1], image_id[2], image_id)


def main():
    import pandas as pd
    import torch
    from sklearn.model_selection import StratifiedKFold
    train = pd.read_pickle("train.pkl")
    train["file_path"] = train["image_id"].apply(getTrainFilePath)
    print(train["file_path"])
    print(f"tokenizer.stoi: {CONFIG.TOKENIZER.stoi}")
    print(train.keys())
    print(train["file_path"][0])
    folds = train.copy()
    fold = StratifiedKFold(
        n_splits=CONFIG.NUMBER_OF_FOLDS, shuffle=True, random_state=42)
    for n, (train_index, val_index) in enumerate(
            fold.split(folds, folds["InChI_length"])):
        folds.loc[val_index, "fold"] = int(n)
    folds["fold"] = folds["fold"].astype(int)
    print(folds.groupby(["fold"]).size())
    train_transforms = Compose([
        Resize(CONFIG.PATCH_SIZE, CONFIG.PATCH_SIZE),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    valid_transforms = Compose([
        Resize(CONFIG.PATCH_SIZE, CONFIG.PATCH_SIZE),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    valid_fold_index = 0
    trn_idx = folds[folds['fold'] != valid_fold_index].index
    val_idx = folds[folds['fold'] == valid_fold_index].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds['InChI'].values
    train_dataset = Dataset(train_folds, CONFIG.TOKENIZER, train_transforms)
    valid_dataset = Dataset(valid_folds, CONFIG.TOKENIZER, valid_transforms)
    learner = Learner(train_dataset, valid_dataset)
    learner.train()


main()
