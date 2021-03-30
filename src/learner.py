import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import trange

# Project files
from config import CONFIG
import src.callbacks as cb
import src.utils as ut


class Learner():
    def __init__(self, train_dataset, valid_dataset):
        self.epoch_metrics = {}

        # Load from config
        self.tokenizer = CONFIG.TOKENIZER
        self.epochs = CONFIG.EPOCHS
        self.encoder = CONFIG.ENCODER.to(CONFIG.DEVICE)
        self.decoder = CONFIG.DECODER.to(CONFIG.DEVICE)
        self.encoder_optimizer = CONFIG.ENCODER_OPTIMIZER
        self.decoder_optimizer = CONFIG.DECODER_OPTIMIZER
        self.encoder_scheduler = CONFIG.ENCODER_SCHEDULER
        self.decoder_scheduler = CONFIG.DECODER_SCHEDULER
        self.loss_function = CONFIG.LOSS_FUNCTION

        # Callbacks
        self.start_epoch, self.model_directory, validation_loss_min = \
            ut.loadModel(
                [self.encoder, self.decoder], self.epoch_metrics,
                CONFIG.MODEL_PATH,
                [self.encoder_optimizer, self.decoder_optimizer],
                CONFIG.LOAD_MODEL)
        self.csv_logger = cb.CsvLogger(self.model_directory)
        self.early_stopping = cb.EarlyStopping(
            self.model_directory, CONFIG.PATIENCE,
            validation_loss_min=validation_loss_min)

        # Train and validation batch generators
        self.train_dataloader = ut.getDataloader(train_dataset)
        self.valid_dataloader = ut.getDataloader(valid_dataset, shuffle=False)
        self.number_of_train_batches = ut.getIterations(self.train_dataloader)
        self.number_of_valid_batches = len(self.valid_dataloader)

    def logData(self):
        self.csv_logger.__call__(self.epoch_metrics)
        self.early_stopping.__call__(
            self.epoch_metrics,
            [self.encoder, self.decoder],
            [self.encoder_optimizer, self.decoder_optimizer])
        self.encoder_scheduler.step(self.epoch_metrics["valid_loss"])
        self.decoder_scheduler.step(self.epoch_metrics["valid_loss"])
        ut.saveLearningCurve(model_directory=self.model_directory)

    def validationIteration(self, batch, i):
        images, labels, label_lengths = ut.toDevice(batch)
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
        ut.updateEpochMetrics(
            text_prediction, text_target, loss, i,
            self.epoch_metrics, "valid", self.encoder_optimizer)

    def trainIteration(self, batch, i):
        # Feed forward and backpropagation
        images, labels, label_lengths = ut.toDevice(batch)
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
            ut.updateEpochMetrics(
                text_prediction, text_target, loss, i,
                self.epoch_metrics, "train")

    def validationEpoch(self):
        print()
        progress_bar = trange(self.number_of_valid_batches, leave=False)
        progress_bar.set_description(" Validation")
        for i, batch in zip(progress_bar, self.valid_dataloader):
            self.validationIteration(batch, i)
        print("\n{}".format(ut.getProgressbarText(
            self.epoch_metrics, "Valid")))
        self.logData()

    def trainEpoch(self, epoch):
        progress_bar = trange(self.number_of_train_batches, leave=False)
        progress_bar.set_description(f" Epoch {epoch}/{self.epochs}")
        for i, batch in zip(progress_bar, self.train_dataloader):

            # Validation epoch before last batch
            if i == self.number_of_train_batches - 1:
                with torch.no_grad():
                    self.validationEpoch()
            self.trainIteration(batch, i)
            with torch.no_grad():
                progress_bar.display(
                    ut.getProgressbarText(self.epoch_metrics, "Train"), 1)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs+1):
            if self.early_stopping.isEarlyStop():
                break
            self.epoch_metrics = ut.initializeEpochMetrics(epoch)
            self.trainEpoch(epoch)
