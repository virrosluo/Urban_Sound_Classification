from dataloader import UrbanSoundLightning
from model import ResnetLightning, CNNNetwork

import lightning
from lightning.pytorch.loggers.wandb import WandbLogger
import torchvision

NUM_CLASSES = 10
ANNOTATIONS_FILE = "./data/metadata/UrbanSound8K.csv"
AUDIO_DIR = "./data/audio"
SAMPLE_RATE = 22050
TARGET_LENGTH = 22050

TRAIN_BATCH_SIZE = 128
INFER_BATCH_SIZE = 256

lightning_model = ResnetLightning(
    model=CNNNetwork(), 
    num_classes=NUM_CLASSES
)

lightning_dataset = UrbanSoundLightning(
    annotations_file=ANNOTATIONS_FILE,
    audio_dir=AUDIO_DIR,
    train_batch_size=TRAIN_BATCH_SIZE,
    infer_batch_size=INFER_BATCH_SIZE,
    target_sample_rate=SAMPLE_RATE,
    target_length=TARGET_LENGTH
)

logger = WandbLogger(
    name="UrbanSound", 
    save_dir="./cache", 
    project="Urban Sound Classification",
    log_model=True
)

trainer = lightning.Trainer(
    logger=logger,
    max_epochs=100,
    val_check_interval=0.2,
    log_every_n_steps=1,
    default_root_dir="./cache",
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm"
)

trainer.fit(
    model=lightning_model,
    datamodule=lightning_dataset
)

print(trainer.test(
    model=lightning_model,
    datamodule=lightning_dataset
))