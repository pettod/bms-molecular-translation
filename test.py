import os

import numpy as np
import pandas as pd
import torch
from albumentations import (Blur, Compose, Cutout, HorizontalFlip,
                            IAAAdditiveGaussianNoise, Normalize, OneOf,
                            RandomBrightness, RandomBrightnessContrast,
                            RandomContrast, RandomCrop, RandomResizedCrop,
                            Resize, Rotate, ShiftScaleRotate, Transpose,
                            VerticalFlip)
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CONFIG
from src.dataset import TestDataset as Dataset
from src.utils import loadModel


# Data paths
TEST_X_DIR = os.path.join(CONFIG.DATA_ROOT, "test")
MODEL_PATH = "saved_models/2021-04-06_212914_first_train/model.pt"
TOKENIZER_PATH = "tokenizer.pth"
SAMPLE_SUBMISSION_PATH = os.path.join(CONFIG.DATA_ROOT, "sample_submission.csv")


def getTestFilePath(image_id):
    return "{}/{}/{}/{}/{}.png".format(
        TEST_X_DIR, image_id[0], image_id[1], image_id[2], image_id
    )


def main():
    test = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    test["file_path"] = test["image_id"].apply(getTestFilePath)
    tokenizer = torch.load(TOKENIZER_PATH)
    transforms = Compose([
        Resize(CONFIG.PATCH_SIZE, CONFIG.PATCH_SIZE),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    test_dataset = Dataset(test, transforms)
    test_dataloader = DataLoader(
        test_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False,
        num_workers=CONFIG.NUMBER_OF_DATALOADER_WORKERS)

    # Predict and save
    with torch.no_grad():
        encoder = CONFIG.ENCODER.to(CONFIG.DEVICE)
        decoder = CONFIG.DECODER.to(CONFIG.DEVICE)
        loadModel([encoder, decoder], "saved_models", MODEL_PATH)
        text_preds = []
        for i, images in enumerate(tqdm(test_dataloader)):
            images = images.to(CONFIG.DEVICE)
            features = encoder(images)
            predictions = decoder.predict(features, 275, tokenizer)
            predicted_sequence = torch.argmax(predictions, -1).cpu().numpy()
            _text_preds = tokenizer.predict_captions(predicted_sequence)
            text_preds.append(_text_preds)
        text_preds = np.concatenate(text_preds)
        test["InChI"] = [f"InChI=1S/{text}" for text in text_preds]
        test[["image_id", "InChI"]].to_csv("submission.csv", index=False)
        test[["image_id", "InChI"]].head()


main()
