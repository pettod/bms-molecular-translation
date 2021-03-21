import numpy as np
import torch
import os
import pandas as pd


def createSaveModelDirectory(save_directory):
    # Create folders if do not exist
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a 
    given patience.
    """
    def __init__(
            self, save_model_directory, patience=10, verbose=False, delta=0,
            counter=0, validation_loss_min=np.Inf):
        """
        Args:
            patience : int
                How long to wait after last time validation loss improved.
            verbose : bool
                If True, prints a message for each validation loss improvement. 
            delta : float
                Minimum change in the monitored quantity to qualify as an
                improvement.
            counter : int
                Early stopping counter
            validation_loss_min : float
                Minimum validation loss achieved
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = validation_loss_min
        self.delta = delta
        self.save_directory = save_model_directory

    def __call__(self, epoch_metrics, model, optimizer):
        createSaveModelDirectory(self.save_directory)
        val_loss = epoch_metrics["valid_loss"]
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_metrics, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_metrics, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, epoch_metrics, model, optimizer):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print("Validation loss decreased. Model saved")
        torch.save({
            **{
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            },
            **epoch_metrics},
            os.path.join(self.save_directory, "model.pt"))
        self.val_loss_min = epoch_metrics["valid_loss"]

    def isEarlyStop(self):
        if self.early_stop:
            print("Early stop")
        return self.early_stop


class CsvLogger:
    def __init__(self, save_model_directory):
        self.save_directory = save_model_directory
        self.logs_file_path = os.path.join(
            save_model_directory, "logs.csv")

    def __call__(self, loss_and_metrics):
        createSaveModelDirectory(self.save_directory)

        # Create CSV file
        new_data_frame = pd.DataFrame(loss_and_metrics, index=[0])
        if not os.path.isfile(self.logs_file_path):
            new_data_frame.to_csv(
                self.logs_file_path, header=True, index=False)
        else:
            with open(self.logs_file_path, 'a') as old_data_frame:
                new_data_frame.to_csv(
                    old_data_frame, header=False, index=False)
