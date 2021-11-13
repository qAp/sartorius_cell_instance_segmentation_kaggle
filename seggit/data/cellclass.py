
import os
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.crops.transforms import RandomCrop
import pytorch_lightning as pl
import cv2

from seggit.data.util import *
from seggit.data.util import generate_kfold



DIR_IMG = f'{DIR_BASE}/train'
DIR_MASK = '/kaggle/input/sardata-train-mask/train_mask'

FOLD = 0
IMAGE_SIZE = 480
BATCH_SIZE = 4
NUM_WORKERS = 0


class CellClassDataset(torch.utils.data.Dataset):

    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.transform = transform

        self.imgids = df['id'].unique()

    def __len__(self):
        return len(self.imgids)

    def __getitem__(self, idx):
        imgid = self.imgids[idx]
        img = cv2.imread(f'{DIR_IMG}/{imgid}.png')
        mask = cv2.imread(f'{DIR_MASK}/{imgid}.png')

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        img = (img / 255)[[0], ...]
        mask = mask[..., 0].type(torch.float32)
        return img, mask


class CellClass(pl.LightningDataModule):

    def __init__(self, args=None):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.fold = self.args.get('fold', FOLD)
        self.image_size = self.args.get('image_size', IMAGE_SIZE)
        self.batch_size = self.args.get('batch_size', BATCH_SIZE)
        self.num_workers = self.args.get('num_workers', NUM_WORKERS)
        self.on_gpu = isinstance(self.args.get('gpus', None), (int, list))

        tfms_default = [
            RandomCrop(height=self.image_size, 
                       width=self.image_size, 
                       always_apply=True),
            ToTensorV2()
            ]
        self.transform = A.Compose(tfms_default)

        self.train_ds: torch.utils.data.Dataset
        self.valid_ds: torch.utils.data.Dataset

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--fold', type=int, default=FOLD)
        parser.add_argument('--image_size', type=int, default=IMAGE_SIZE)
        parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
        parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)

    def prepare_data(self):
        if not os.path.exists(DIR_MASK):
            print(
                f'Masks are not available at {DIR_MASK}.  ' 
                'Generate them with `python seggit/data/generate_mask.py`.'
            )

        if not os.path.exists(DIR_KFOLD):
            generate_kfold()


    def setup(self):
        try:
            train_df = pd.read_csv(f'{DIR_KFOLD}/train_fold{self.fold}.csv')
            valid_df = pd.read_csv(f'{DIR_KFOLD}/valid_fold{self.fold}.csv')
        except FileNotFoundError:
            print(f'Make sure folds csv files are available at {DIR_KFOLD},\n'
                  f'or use {self.prepare_data} to generate these first.')
            raise

        self.train_ds = CellClassDataset(train_df, transform=self.transform)
        self.valid_ds = CellClassDataset(valid_df, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_ds, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )




