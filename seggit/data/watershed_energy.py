
import os, sys
import argparse
import numpy as np
import pandas as pd
from seggit.data.instance_direction import FOLD, IMAGE_SIZE, NUM_WORKERS
import torch
import cv2
import albumentations as albu
import pytorch_lightning as pl

from seggit.data.config import DIR_KFOLD, DIR_MASK, DIR_AREA, DIR_DTFM, DIR_ENERGY


FOLD = 0
IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 0



class WatershedEnergyDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()

        self.df = df
        self.transform = transform

        self.imgids = df['id'].unique()

    def __len__(self):
        return len(self.imgids)

    def __getitem__(self, i):
        imgid = self.imgids[i]

        semseg = cv2.imread(f'{DIR_MASK}/{imgid}.png')
        instance_area = np.load(f'{DIR_AREA}/{imgid}.npy')
        dtfm = np.load(f'{DIR_DTFM}/{imgid}.npy')
        energy = np.load(f'{DIR_ENERGY}/{imgid}.npy')

        mask = np.stack(
            [
                semseg[:, :, 0].astype(np.float32),
                instance_area[:, :, 0].astype(np.float32),
                dtfm[:, :].astype(np.float32),
                energy[:, :].astype(np.float32)
            ],
            axis=2)

        if self.transform:
            transformed = self.transform(
                image=mask[...,[0]].astype(np.uint8),  # dummy image
                mask=mask)
            mask = transformed['mask']

        grad_dtfm = np.stack(np.gradient(mask[..., 2]), axis=2)
        norm = np.linalg.norm(grad_dtfm, axis=2, keepdims=True)
        uvec = np.nan_to_num(grad_dtfm / norm)

        semseg = mask[..., [0]]
        area = mask[..., [1]]
        energy = mask[..., [3]]

        return semseg, area, uvec, energy


class WatershedEnergy(pl.LightningDataModule):
    def __init__(self, args=None):
        super().__init__()

        self.args = vars(args) if args is not None else {}

        self.fold = self.args.get('fold', FOLD)
        self.image_size = self.args.get('image_size', IMAGE_SIZE)
        self.batch_size = self.args.get('batch_size', BATCH_SIZE)
        self.num_workers = self.args.get('num_workers', NUM_WORKERS)
        self.on_gpu = isinstance(self.args.get('gpu', None), (int, str))

        transform = _tfms(self.image_size)
        self.transform = albu.Compose(transform)

        self.train_ds: WatershedEnergyDataset
        self.valid_ds: WatershedEnergyDataset

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--fold', type=int, default=FOLD)
        parser.add_argument('--image_size', type=int, default=IMAGE_SIZE)
        parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
        parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)

    def prepare_data(self):
        assert os.path.exists(DIR_KFOLD), (
            f'K-folds directory {DIR_KFOLD} not found.'
            f'Generate k-folds with data.util.generate_kfold'
        )

        assert os.path.exists(DIR_ENERGY), (
            f'Watershed energy {DIR_ENERGY} not found.'
            'Load this or generate with '
            'data/scripts/generate_watershed_energy.py'
        )

        assert os.path.exists(DIR_DTFM), (
            f'Distance transform {DIR_DTFM} not found.'
            'Load or generate with '
            'data/scripts/generate_normalised_gradient.py'
        )

        assert os.path.exists(DIR_MASK), (
            f'Semantic segmentation {DIR_MASK} not found.'
            'Load this or generate with '
            'data/scripts/generate_mask.py'
        )

        assert os.path.exists(DIR_AREA), (
            f'Instance area {DIR_AREA} not found.'
            'Load this or generate with'
            'data/scripts/generate_instance_area.py'
        )

    def setup(self):
        df_train = pd.read_csv(f'{DIR_KFOLD}/train_fold{self.fold}.csv')
        df_valid = pd.read_csv(f'{DIR_KFOLD}/valid_fold{self.fold}.csv')

        self.train_ds = WatershedEnergyDataset(df_train, transform=self.transform)
        self.valid_ds = WatershedEnergyDataset(df_valid, transform=self.transform)

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
