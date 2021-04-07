import os
import torch
import torch.optim as optim
from multiprocessing import cpu_count
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from src.loss_functions import costFunction as lossFunction
from src.network import Encoder, DecoderWithAttention


class CONFIG:
    # Paths
    DATA_ROOT = os.path.realpath("../input/bms-molecular-translation")
    TRAIN_X_DIR = os.path.join(DATA_ROOT, "train")
    TRAIN_Y_FIL = os.path.join(DATA_ROOT, "train_labels.csv")

    # General parameters
    EPOCHS = 1000
    LOAD_MODEL = False
    MODEL_PATH = None
    DROP_LAST_BATCH = False
    NUMBER_OF_DATALOADER_WORKERS = cpu_count()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    LOSS_FUNCTION = lossFunction
    BATCH_SIZE = 64
    PATCH_SIZE = 224
    PATIENCE = 10
    LEARNING_RATES = [1e-4, 4e-4]
    ITERATIONS_PER_EPOCH = 1000
    NUMBER_OF_FOLDS = 50

    # Model
    TOKENIZER = torch.load("tokenizer.pth")
    ENCODER = Encoder("resnet34", pretrained=True)
    ENCODER_OPTIMIZER = optim.Adam(ENCODER.parameters(), lr=LEARNING_RATES[0])
    ENCODER_SCHEDULER = CosineAnnealingLR(
        ENCODER_OPTIMIZER, T_max=4, eta_min=1e-6, last_epoch=-1)
    DECODER = DecoderWithAttention(
        attention_dim=256,
        embed_dim=256,
        decoder_dim=512,
        vocab_size=len(TOKENIZER),
        dropout=0.5,
        device=DEVICE
    )
    DECODER_OPTIMIZER = optim.Adam(
        DECODER.parameters(), lr=LEARNING_RATES[1], weight_decay=1e-6)
    DECODER_SCHEDULER = CosineAnnealingLR(
        DECODER_OPTIMIZER, T_max=4, eta_min=1e-6, last_epoch=-1)
