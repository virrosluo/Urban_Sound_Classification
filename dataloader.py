import os
from typing import *
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import multiprocessing
import pandas as pd
import torchaudio
import lightning

class UrbanSoundLightning(lightning.LightningDataModule):
    def __init__(
        self, 
        annotations_file, 
        audio_dir, 
        train_batch_size,
        infer_batch_size,
        target_sample_rate = 22050, 
        target_length = 22050):
        super().__init__()
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.target_length = target_length
        self.target_sample_rate = target_sample_rate
        self.train_batch_size = train_batch_size
        self.infer_batch_size = infer_batch_size

    def prepare_data(self):
        self.annotations["path"] = self.annotations.apply(
            lambda sample: os.path.join(self.audio_dir, f"fold{sample['fold']}", sample["slice_file_name"]),
            axis=1
        )

    def setup(self, stage):
        transformation = [
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.target_sample_rate,
                n_fft=1024,
                hop_length=512,
                n_mels=64
            ),
            torchaudio.transforms.AmplitudeToDB(
                stype='power',
                top_db=80,
            )
        ]
        dataset = LazyUrbanSoundDataset(
            annotations=self.annotations,
            transformation=transformation,
            target_sample_rate=self.target_sample_rate,
            target_length=self.target_length
        )

        self.train_dataset, self.valid_dataset, self.test_dataset = random_split(dataset, [0.8, 0.1, 0.1])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=1
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.infer_batch_size,
            shuffle=False,
            num_workers=1
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.infer_batch_size,
            shuffle=False,
            num_workers=1
        )

class LazyUrbanSoundDataset(Dataset):
    def __init__(self, annotations: pd.DataFrame, transformation, target_sample_rate, target_length):
        self.transformation = transformation
        self.target_length = target_length
        self.target_sample_rate = target_sample_rate
        self.annotations = annotations
        self.cache = dict()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]

        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)

        signal = self._resample_if_necessary (signal, sr)
        signal = self._padding_if_neccessary(signal)
        signal = self._mix_down_if_necessary(signal)
        signal = self._get_transformation(signal)

        self.cache[index] = (signal, label)
        return signal, label

    def _padding_if_neccessary(self, signal):
        if signal.shape[-1] > self.target_length:
            return signal[..., :self.target_length]
        elif signal.shape[-1] < self.target_length:
            return torch.nn.functional.pad(signal, (0, self.target_length - signal.shape[-1]))
        else:
            return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        return torch.mean(signal, dim=0, keepdim=True) if signal.shape[0] > 1 else signal

    def _get_audio_sample_path(self, index):
        return self.annotations["path"][index]

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]

    def _get_transformation(self, signal):
        for transform in self.transformation:
            signal = transform(signal)
        return signal


if __name__ == "__main__":
    ANNOTATIONS_FILE = "./data/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "./data/audio"
    SAMPLE_RATE = 22050
    TARGET_LENGTH = 22050
    
    lightning_dataset = UrbanSoundLightning(
        annotations_file=ANNOTATIONS_FILE, 
        audio_dir=AUDIO_DIR,
        train_batch_size=32,
        infer_batch_size=64,
        target_sample_rate=SAMPLE_RATE,
        target_length=TARGET_LENGTH
    )