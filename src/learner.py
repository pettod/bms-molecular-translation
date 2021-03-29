import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

import os
from tqdm import trange

# Project files
from src.callbacks import CsvLogger, EarlyStopping
from src.network import Encoder, DecoderWithAttention
from src.utils import \
    initializeEpochMetrics, updateEpochMetrics, getProgressbarText, \
    saveLearningCurve, loadModel, getTorchDevice, bmsCollate


class Learner():
    def __init__(
            self, train_dataset, valid_dataset, batch_size, learning_rates,
            loss_function, tokenizer, patience=10, num_workers=1,
            load_pretrained_weights=False, model_path=None,
            drop_last_batch=False):
        self.device = getTorchDevice()
        self.epoch_metrics = {}
        self.tokenizer = tokenizer

        # Model, optimizer, loss function, scheduler
        self.encoder = Encoder("resnet34", pretrained=True).to(self.device)
        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=learning_rates[0])
        self.start_epoch, self.model_directory, validation_loss_min = \
            loadModel(
                self.encoder, self.epoch_metrics, "saved_models",
                model_path, self.encoder_optimizer, load_pretrained_weights)
        self.encoder_scheduler = ReduceLROnPlateau(
            self.encoder_optimizer, "min", 0.3, patience//3, min_lr=1e-8)
        self.decoder = DecoderWithAttention(
            attention_dim=256,
            embed_dim=256,
            decoder_dim=512,
            vocab_size=len(tokenizer),
            dropout=0.5,
            device=self.device
        ).to(self.device)
        self.decoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=learning_rates[1])
        self.start_epoch, self.model_directory, validation_loss_min = \
            loadModel(
                self.decoder, self.epoch_metrics, "saved_models",
                model_path, self.decoder_optimizer, load_pretrained_weights)
        self.decoder_scheduler = ReduceLROnPlateau(
            self.decoder_optimizer, "min", 0.3, patience//3, min_lr=1e-8)

        # Callbacks
        self.loss_function = loss_function
        self.csv_logger = CsvLogger(self.model_directory)
        self.early_stopping = EarlyStopping(
            self.model_directory, patience,
            validation_loss_min=validation_loss_min)

        # Train and validation batch generators
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=drop_last_batch,
            pin_memory=True, collate_fn=bmsCollate)
        self.valid_dataloader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=drop_last_batch,
            pin_memory=True, collate_fn=bmsCollate)
        self.number_of_train_batches = len(self.train_dataloader)
        self.number_of_valid_batches = len(self.valid_dataloader)

    def validationEpoch(self):
        print()
        progress_bar = trange(self.number_of_valid_batches, leave=False)
        progress_bar.set_description(" Validation")

        # Run batches
        for i, (images, labels, label_lengths) in zip(
                progress_bar, self.valid_dataloader):
            images, labels, label_lengths = \
                    images.to(self.device), labels.to(self.device), \
                    label_lengths.to(self.device)
            features = self.encoder(images)
            predictions, caps_sorted, decode_lengths, alphas, sort_ind = \
                self.decoder(features, labels, label_lengths)
            targets = caps_sorted[:, 1:]
            predictions = pack_padded_sequence(
                predictions, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True).data
            loss = self.loss_function(predictions, targets)
            predicted_sequence = torch.argmax(
                predictions, -1).cpu().numpy()
            text_prediction = self.tokenizer.predict_captions(
                [predicted_sequence])[0]
            text_target = self.tokenizer.predict_captions(
                [targets.cpu().numpy()])[0]
            updateEpochMetrics(
                text_prediction, text_target, loss, i,
                self.epoch_metrics, "valid", self.encoder_optimizer)

        # Logging
        print("\n{}".format(getProgressbarText(self.epoch_metrics, "Valid")))
        self.csv_logger.__call__(self.epoch_metrics)
        self.early_stopping.__call__(
            self.epoch_metrics, self.encoder, self.encoder_optimizer)
        self.encoder_scheduler.step(self.epoch_metrics["valid_loss"])
        self.decoder_scheduler.step(self.epoch_metrics["valid_loss"])
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
            for i, (images, labels, label_lengths) in zip(
                    progress_bar, self.train_dataloader):

                # Validation epoch before last batch
                if i == self.number_of_train_batches - 1:
                    with torch.no_grad():
                        self.validationEpoch()

                # Feed forward and backpropagation
                images, labels, label_lengths = \
                    images.to(self.device), labels.to(self.device), \
                    label_lengths.to(self.device)
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                features = self.encoder(images)
                predictions, caps_sorted, decode_lengths, alphas, sort_ind = \
                    self.decoder(features, labels, label_lengths)
                targets = caps_sorted[:, 1:]
                predictions = pack_padded_sequence(
                    predictions, decode_lengths, batch_first=True).data
                targets = pack_padded_sequence(
                    targets, decode_lengths, batch_first=True).data
                loss = self.loss_function(predictions, targets)
                loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                # Compute metrics
                with torch.no_grad():
                    predicted_sequence = torch.argmax(
                        predictions, -1).cpu().numpy()
                    text_prediction = self.tokenizer.predict_captions(
                        [predicted_sequence])[0]
                    text_target = self.tokenizer.predict_captions(
                        [targets.cpu().numpy()])[0]
                    updateEpochMetrics(
                        text_prediction, text_target, loss, i,
                        self.epoch_metrics, "train")
                    progress_bar.display(
                        getProgressbarText(self.epoch_metrics, "Train"), 1)
