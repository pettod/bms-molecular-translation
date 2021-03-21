import os
from torchvision import transforms
from multiprocessing import cpu_count

# Project files
from src.dataset import ImageDataset as Dataset
from src.learner import Learner
from src.loss_functions import maeGradientPlusMae as lossFunction

# Data paths
DATA_ROOT = os.path.realpath("../bms-molecular-translation-data")
TRAIN_X_DIR = os.path.join(DATA_ROOT, "train")
TRAIN_Y_FIL = os.path.join(DATA_ROOT, "train_labels.csv")
VALID_X_DIR = TRAIN_X_DIR
VALID_Y_FIL = TRAIN_Y_FIL

# Model parameters
LOAD_MODEL = False
MODEL_PATH = None
BATCH_SIZE = 16
PATCH_SIZE = 256
PATIENCE = 10
LEARNING_RATE = 1e-4
DROP_LAST_BATCH = False
NUMBER_OF_DATALOADER_WORKERS = 0 #cpu_count()


def main():
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]),
    ])
    valid_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]),
    ])
    train_dataset = Dataset(TRAIN_X_DIR, TRAIN_Y_FIL, train_transforms)
    valid_dataset = Dataset(VALID_X_DIR, VALID_Y_FIL, valid_transforms)
    learner = Learner(
        train_dataset, valid_dataset, BATCH_SIZE, LEARNING_RATE, lossFunction,
        PATIENCE, NUMBER_OF_DATALOADER_WORKERS, LOAD_MODEL, MODEL_PATH,
        DROP_LAST_BATCH)
    learner.train()


main()
