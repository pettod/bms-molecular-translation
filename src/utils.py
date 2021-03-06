import glob
from inspect import getmembers, isfunction
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import time
import torch
import torch.nn as nn
from math import ceil
from torch.nn.utils.rnn import pad_sequence

# Project files
from config import CONFIG
import src.metrics as metrics


def getMetrics():
    metrics_name_and_function_pointers = [
        metric for metric in getmembers(metrics, isfunction)
        if metric[1].__module__ == metrics.__name__]
    return metrics_name_and_function_pointers


def computeMetrics(y_pred, y_true):
    metric_functions = getMetrics()
    metric_scores = {}
    for metric_name, metric_function_pointer in metric_functions:
        metric_scores[metric_name] = metric_function_pointer(y_pred, y_true)
    return metric_scores


def initializeEpochMetrics(epoch):
    metric_functions = getMetrics()
    epoch_metrics = {}
    epoch_metrics["epoch"] = epoch
    epoch_metrics["train_loss"] = 0
    epoch_metrics["valid_loss"] = 0
    for metric_name, _ in metric_functions:
        epoch_metrics[f"train_{metric_name}"] = 0
        epoch_metrics[f"valid_{metric_name}"] = 0
    return epoch_metrics


def updateEpochMetrics(
        y_pred, y_true, loss, epoch_iteration_index, epoch_metrics, mode,
        optimizer=None):
    metric_scores = computeMetrics(y_pred, y_true)
    metric_scores["loss"] = loss
    for key, value in metric_scores.items():
        if type(value) == torch.Tensor:
            value = value.item()

        epoch_metrics[f"{mode}_{key}"] += ((
            value - epoch_metrics[f"{mode}_{key}"]) /
            (epoch_iteration_index + 1))
    if optimizer:
        epoch_metrics["learning_rate"] = optimizer.param_groups[0]["lr"]


def getProgressbarText(epoch_metrics, mode):
    text = f" {mode}:"
    mode = mode.lower()
    for key, value in epoch_metrics.items():
        if mode not in key:
            continue
        text += " {}: {:2.4f}.".format(key.replace(f"{mode}_", ""), value)
    return text


def saveLearningCurve(
        log_file_path=None, model_directory=None, model_root="saved_models",
        xticks_limit=13):
    # Read CSV log file
    if log_file_path is None and model_directory is None:
        log_file_path = sorted(glob.glob(os.path.join(
            model_root, *['*', "*.csv"])))[-1]
    elif model_directory is not None:
        log_file_path = glob.glob(os.path.join(
            model_directory, "*.csv"))[0]
    log_file = pd.read_csv(log_file_path)

    # Read data into dictionary
    log_data = {}
    for column in log_file:
        if column == "epoch":
            log_data[column] = np.array(log_file[column].values, dtype=np.str)
        elif column == "learning_rate":
            continue
        else:
            log_data[column] = np.array(log_file[column].values)
    number_of_epochs = log_file.shape[0]

    # Remove extra printings of same epoch
    used_xticks = [i for i in range(number_of_epochs)]
    epoch_string_data = []
    previous_epoch = -1
    for i, epoch in enumerate(reversed(log_data["epoch"])):
        if epoch != previous_epoch:
            epoch_string_data.append(epoch)
        else:
            used_xticks.pop(-1*i - 1)
        previous_epoch = epoch
    epoch_string_data = epoch_string_data[::-1]
    log_data.pop("epoch", None)

    # Limit number of printed epochs in x axis
    used_xticks = used_xticks[::ceil(number_of_epochs / xticks_limit)]
    epoch_string_data = epoch_string_data[::ceil(
        number_of_epochs / xticks_limit)]

    # Define train and validation subplots
    figure_dict = {}
    for key in log_data.keys():
        metric = key.split('_')[-1]
        if metric not in figure_dict:
            figure_dict[metric] = len(figure_dict.keys()) + 1
    number_of_subplots = len(figure_dict.keys())

    # Save learning curves plot
    plt.figure(figsize=(15, 7))
    import warnings
    warnings.filterwarnings("ignore")
    for i, key in enumerate(log_data.keys()):
        metric = key.split('_')[-1]
        plt.subplot(1, number_of_subplots, figure_dict[metric])
        plt.plot(range(number_of_epochs), log_data[key], label=key)
        plt.xticks(used_xticks, epoch_string_data)
        plt.xlabel("Epoch")
        plt.title(metric.upper())
        plt.legend()
    plt.tight_layout()
    plt.savefig("{}.{}".format(log_file_path.split('.')[0], "png"))


def loadModel(
        model, epoch_metrics, model_path=None, optimizer=None,
        load_pretrained_weights=True, model_root="saved_models"):
    if type(model) != list:
        model = [model]
    for i in range(len(model)):
        print("{:,} model parameters".format(
            sum(p.numel() for p in model[i].parameters() if p.requires_grad)))
    validation_loss_min = np.Inf
    start_epoch = 1
    model_directory = os.path.join(
        model_root, time.strftime("%Y-%m-%d_%H%M%S"))
    if len(sys.argv) > 1 and sys.argv[1] != '&':
        model_directory = f"{model_directory}_{sys.argv[1]}"

    if load_pretrained_weights:

        # Load latest model
        if model_path is None:
            model_name = sorted(glob.glob(os.path.join(
                model_root, *['*', "*.pt"])))[-1]
        else:

            # Load model based on index
            if type(model_path) == int:
                model_name = sorted(glob.glob(os.path.join(
                    model_root, *['*', "*.pt"])))[model_path]

            # Load defined model path
            else:
                model_name = model_path

        model_directory = os.path.join(*model_name.split('/')[:-1])
        checkpoint = torch.load(model_name)
        for i in range(len(model)):
            model[i].load_state_dict(checkpoint[f"model_{i}"])
            model[i].eval()
        if optimizer:
            if type(optimizer) != list:
                optimizer = [optimizer]
            for i in range(len(optimizer)):
                optimizer[i].load_state_dict(checkpoint[f"optimizer_{i}"])
        validation_loss_min = checkpoint["valid_loss"]
        log_files = glob.glob(os.path.join(model_directory, "*.csv"))
        if len(log_files):
            start_epoch = int(pd.read_csv(
                log_files[0])["epoch"].to_list()[-1]) + 1
        print("Loaded model: {}".format(model_name))

    return start_epoch, model_directory, validation_loss_min


def getDataloader(dataset, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=shuffle,
        num_workers=CONFIG.NUMBER_OF_DATALOADER_WORKERS,
        drop_last=CONFIG.DROP_LAST_BATCH, pin_memory=True,
        collate_fn=bmsCollate)


def getIterations(data_loader):
    if CONFIG.ITERATIONS_PER_EPOCH > 1:
        return min(len(data_loader), CONFIG.ITERATIONS_PER_EPOCH)
    elif CONFIG.ITERATIONS_PER_EPOCH == 1:
        return len(data_loader)
    else:
        return int(len(data_loader) * CONFIG.ITERATIONS_PER_EPOCH)


def toDevice(batch):
    if type(batch) == tuple:
        batch = tuple([sample.to(CONFIG.DEVICE) for sample in batch])
    elif type(batch) == list:
        batch = [sample.to(CONFIG.DEVICE) for sample in batch]
    else:
        batch = batch.to(CONFIG.DEVICE)
    return batch


def bmsCollate(batch):
    imgs, labels, label_lengths = [], [], []
    for data_point in batch:
        imgs.append(data_point[0])
        labels.append(data_point[1])
        label_lengths.append(data_point[2])
    labels = pad_sequence(
        labels, batch_first=True, padding_value=CONFIG.TOKENIZER.stoi["<pad>"])
    return (
        torch.stack(imgs),
        labels,
        torch.stack(label_lengths).reshape(-1, 1)
    )
