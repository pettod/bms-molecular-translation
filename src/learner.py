import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import os
from tqdm import trange

# Project files
from src.callbacks import CsvLogger, EarlyStopping
from src.network import Net
from src.utils import \
    initializeEpochMetrics, updateEpochMetrics, getProgressbarText, \
    saveLearningCurve, loadModel, getTorchDevice


class Learner():
    def __init__(
            self, train_dataset, valid_dataset, batch_size, learning_rate,
            loss_function, patience=10, num_workers=1,
            load_pretrained_weights=False, model_path=None,
            drop_last_batch=False):
        self.device = getTorchDevice()
        self.epoch_metrics = {}

        # Model, optimizer, loss function, scheduler
        self.model = nn.DataParallel(Net()).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_function = loss_function
        self.start_epoch, self.model_directory, validation_loss_min = \
            loadModel(
                self.model, self.epoch_metrics, "saved_models", model_path,
                self.optimizer, load_pretrained_weights)

        # Callbacks
        self.csv_logger = CsvLogger(self.model_directory)
        self.early_stopping = EarlyStopping(
            self.model_directory, patience,
            validation_loss_min=validation_loss_min)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", 0.3, patience//3, min_lr=1e-8)

        # Train and validation batch generators
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=drop_last_batch)
        self.valid_dataloader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=drop_last_batch)
        self.number_of_train_batches = len(self.train_dataloader)
        self.number_of_valid_batches = len(self.valid_dataloader)

    def validationEpoch(self):
        print()
        progress_bar = trange(self.number_of_valid_batches, leave=False)
        progress_bar.set_description(" Validation")

        # Run batches
        for i, (X, y) in zip(progress_bar, self.valid_dataloader):
            X, y = X.to(self.device), y.to(self.device)
            output = self.model(X)
            loss = self.loss_function(output, y)
            updateEpochMetrics(
                output, y, loss, i, self.epoch_metrics, "valid",
                self.optimizer)

        # Logging
        print("\n{}".format(getProgressbarText(self.epoch_metrics, "Valid")))
        self.csv_logger.__call__(self.epoch_metrics)
        self.early_stopping.__call__(
            self.epoch_metrics, self.model, self.optimizer)
        self.scheduler.step(self.epoch_metrics["valid_loss"])
        saveLearningCurve(model_directory=self.model_directory)

    def train(self):
        # Run epochs
        epochs = 1000
        for epoch in range(self.start_epoch, epochs+1):
            if self.early_stopping.isEarlyStop():
                break
            progress_bar = trange(self.number_of_train_batches, leave=False)
            progress_bar.set_description(f" Epoch {epoch}/{epochs}")
            self.epoch_metrics = initializeEpochMetrics(epoch)

            # Run batches
            for i, (X, y) in zip(progress_bar, self.train_dataloader):

                # Validation epoch before last batch
                if i == self.number_of_train_batches - 1:
                    with torch.no_grad():
                        self.validationEpoch()

                # Feed forward and backpropagation
                X, y = X.to(self.device), y.to(self.device)
                self.model.zero_grad()
                output = self.model(X)
                loss = self.loss_function(output, y)
                loss.backward()
                self.optimizer.step()

                # Compute metrics
                with torch.no_grad():
                    updateEpochMetrics(
                        output, y, loss, i, self.epoch_metrics, "train")
                    progress_bar.display(
                        getProgressbarText(self.epoch_metrics, "Train"), 1)
