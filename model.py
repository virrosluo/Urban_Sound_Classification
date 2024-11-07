from torch import nn
from torchmetrics.classification import (
    MulticlassAccuracy, 
    MulticlassPrecision, 
    MulticlassRecall, 
    MulticlassF1Score
)

import wandb
import torch
import lightning
import librosa
import matplotlib.pyplot as plt

class ResnetLightning(lightning.LightningModule):
    def __init__(self, model: nn.Module, num_classes):
        super().__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.accuracy_metric = MulticlassAccuracy(num_classes=num_classes)
        self.precision_metric = MulticlassPrecision(num_classes=num_classes)
        self.recall_metric = MulticlassRecall(num_classes=num_classes)
        self.f1_metric = MulticlassF1Score(num_classes=num_classes)
        
        self.input_dataset = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.input_dataset.append(x)
        logits = self.model(x)
        loss = self.criterion(logits, y)

        self.log_dict({"train_loss": loss})
        
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        x, y = batch
        logits = self.model(x)

        loss = self.criterion(logits, y)
        preds = torch.argmax(torch.softmax(logits, -1), -1)
        
        self.log("valid_loss", loss)
        self.log_dict({
            "valid_accuracy": self.accuracy_metric(preds, y),
            "valid_precision": self.precision_metric(preds, y),
            "valid_recall": self.recall_metric(preds, y),
            "valid_f1": self.f1_metric(preds, y)
        })

        if batch_idx == 0:
            self.log_mel_spectrogram(x, y, preds)

    def test_step(self, batch, batch_idx):
        self.model.eval()
        x, y = batch
        with torch.no_grad():
            logits = self.model(x)
        
        loss = self.criterion(logits, y)
        preds = torch.argmax(torch.softmax(logits, -1), -1)
        
        self.log("test_loss", loss)
        self.log_dict({
            "test_accuracy": self.accuracy_metric(preds, y),
            "test_precision": self.precision_metric(preds, y),
            "test_recall": self.recall_metric(preds, y),
            "test_f1": self.f1_metric(preds, y)
        })
        
        if batch_idx == 0:
            self.log_mel_spectrogram(x, y, preds)

    def log_mel_spectrogram(self, mel_specs: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor):
        """
        Log mel spectrograms to Weights & Biases
        Args:
            mel_specs: Tensor of shape [batch_size, channels, freq_bins, time_steps]
            labels: Ground truth labels
            preds: Model predictions
        """
        num_specs = 3
        # Convert to numpy and reshape
        mel_specs = mel_specs.cpu().numpy()
        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()
        
        # Remove the channel dimension and transpose to get [freq_bins, time_steps]
        mel_specs = mel_specs.squeeze(1)  # Shape becomes [batch_size, freq_bins, time_steps]
        
        fig, axs = plt.subplots(num_specs, 1, figsize=(10, 2 * num_specs))
        if num_specs == 1:
            axs = [axs]
        
        for i in range(min(num_specs, len(mel_specs))):
            spec = mel_specs[i]  # Shape: [freq_bins, time_steps]
            img = librosa.display.specshow(
                spec,
                sr=22050,
                hop_length=512,  # Adjust based on your preprocessing
                x_axis='time',
                y_axis='mel',
                ax=axs[i]
            )
            plt.colorbar(img, ax=axs[i], format='%+2.0f dB')
            axs[i].set_title(f'Mel Spectrogram (Label: {labels[i]}, Pred: {preds[i]})')
        
        plt.tight_layout()
        
        # Log to wandb
        self.logger.experiment.log({
            "mel_spectrograms": wandb.Image(fig),
            "sample_labels": labels[:num_specs].tolist(),
            "sample_predictions": preds[:num_specs].tolist()
        }, commit=False)  # Set commit=False if logging multiple items in one step
        
        plt.close(fig)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=1e-4)
        return optimizer

class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4, 10)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits