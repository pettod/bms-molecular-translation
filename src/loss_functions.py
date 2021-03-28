import torch
import torch.nn as nn


def costFunction(y_pred, y_true):
    return nn.CrossEntropyLoss(ignore_index=192)(y_pred, y_true)
