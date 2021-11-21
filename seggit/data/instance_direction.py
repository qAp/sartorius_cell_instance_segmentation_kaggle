
import os
from typing_extensions import ParamSpecKwargs
import numpy as np
import pandas as pd
import cv2
import torch
import albumentations as albu
import pytorch_lightning as pl
from seggit.data.config import (DIR_KFOLD, 
                                DIR_IMG, 
                                DIR_UVEC, DIR_MASK, DIR_AREA)


FOLD = 0
IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 0


def _default_tfms():
    return [albu.pytorch.transforms.ToTensorV2()]


def _train_tfms(image_size):
    return [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.2,
                              scale_limit=0.3,
                              rotate_limit=180,
                              p=1.,
                              border_mode=0),
        albu.PadIfNeeded(min_height=520,
                         min_width=520,
                         always_apply=True,
                         border_mode=0),
        albu.RandomCrop(height=image_size,
                        width=image_size,
                        always_apply=True),
        albu.GaussNoise(p=1),
        albu.Perspective(p=1),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1)
            ],
            p=0.9),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=5, p=1),
                albu.MotionBlur(blur_limit=5, p=1)
            ],
            p=0.9),
        albu.HueSaturationValue(p=0.9)
    ]


class InstanceDirectionDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()

        self.df = df
        self.transform = transform

        self.imgids = df['id'].unique()

    def __len__(self):
        return len(self.imgids)

    def __getitem__(self, i):
        imgid = self.imgids[i]

        img = cv2.imread(f'{DIR_IMG}/{imgid}.png')
        uvec = np.load(f'{DIR_UVEC}/{imgid}.npy')
        semseg = cv2.imread(f'{DIR_MASK}/{imgid}.png')
        area = np.load(f'{DIR_AREA}/{imgid}.npy')

        semseg = semseg[..., [0]].astype(np.float32)

        mask = np.concatenate([uvec, semseg, area], axis=2)

        if self.transform:
            tfmd = self.transform(image=img, mask=mask)
            img = tfmd['image']
            mask = tfmd['mask']

        uvec = mask[..., :2]
        semseg = mask[..., [2]]
        area = mask[..., [3]]

        img = (img / 255).astype(np.float32)

        return img, uvec, semseg, area


class InstanceDirection(pl.LightningDataModule):
    def __init__(self, args=None):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.fold = self.args.get('fold', FOLD)
        self.image_size = self.args.get('image_size', IMAGE_SIZE)
        self.batch_size = self.args.get('batch_size', BATCH_SIZE)
        self.num_workers = self.args.get('num_workers', NUM_WORKERS)
        self.on_gpu = isinstance(self.args.get('gpus', None) , (int, str))

        self.transform = None

        self.train_ds: InstanceDirectionDataset
        self.valid_ds: InstanceDirectionDataset

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--fold', type=int, default=FOLD)
        parser.add_argument('--image_size', type=int, default=IMAGE_SIZE)
        parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
        parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)

    def prepare_data(self):
        if not os.path.exists(DIR_KFOLD):
            print(f'K-folds directory {DIR_KFOLD} not found.')
            print(f'Generate k-folds with data.util.generate_kfold')

        if not os.path.exists(DIR_IMG):
            print(f'Image directory {DIR_IMG} not found.')
            print('Load competition data.')

        if not os.path.exists(DIR_UVEC):
            print(f'Normalised gradient distance transform '
                  f'{DIR_UVEC} not found.')
            print('Load or generate with '
                  'data/scripts/generate_normalised_gradient.py')

        if not os.path.exists(DIR_MASK):
            print(f'Semantic segmentation {DIR_MASK} not found.')
            print('Load this or generate with ' 
                  'data/scripts/generate_mask.py')

        if not os.path.exists(DIR_AREA):
            print(f'Instance area {DIR_AREA} not found.')
            print('Load this or generate with'
                  'data/scripts/generate_instance_area.py')

    def setup(self):
        try:
            train_df = pd.read_csv(f'{DIR_KFOLD}/train_fold{self.fold}.csv')
            valid_df = pd.read_csv(f'{DIR_KFOLD}/valid_fold{self.fold}.csv')
        except FileNotFoundError:
            print(f'Fold csv files not found at {DIR_KFOLD}')

        self.train_ds = InstanceDirectionDataset(df=train_df, 
                                                 transform=self.transform)
        self.valid_ds = InstanceDirectionDataset(df=valid_df, 
                                                 transform=self.transform)

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
