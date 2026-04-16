import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "horse2zebra/train"
VAL_DIR = "horse2zebra/val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 30
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H = "model/genh.pth.tar"
CHECKPOINT_GEN_Z = "model/genz.pth.tar"
CHECKPOINT_CRITIC_H = "model/critich.pth.tar"
CHECKPOINT_CRITIC_Z = "model/criticz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.1),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
