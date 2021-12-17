
import os, sys
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as albu
import pytorch_lightning as pl

from seggit.data.config import DIR_KFOLD, DIR_IMG, DIR_SEMSEG, DIR_AREA
from seggit.data.config import MEAN_IMAGE, STD_IMAGE
from seggit.data.config import WATERSHED_ENERGY_BINS
from seggit.data.util import semg_to_dtfm, dtfm_to_uvec, dtfm_to_wngy
from seggit.data.transforms import default_tfms, aug_tfms



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

        img = cv2.imread(f'{DIR_IMG}/{imgid}.png')
        semseg = cv2.imread(f'{DIR_SEMSEG}/{imgid}.png')
        area = np.load(f'{DIR_AREA}/{imgid}.npy')

        semseg = semseg.astype(np.float32)

        if self.transform:
            mask = np.concatenate([semseg, area], axis=2)
            tfmd = self.transform(image=img, mask=mask)
            img = tfmd['image']
            mask = tfmd['mask']
            semseg = mask[..., :3]
            area = mask[..., [3]]

        img = (img - MEAN_IMAGE) / STD_IMAGE
        img = img.astype(np.float32)

        semg = semseg[..., [0]] / 255
        dtfm = semg_to_dtfm(semg)
        uvec = dtfm_to_uvec(dtfm)
        wngy = dtfm_to_wngy(dtfm)

        return uvec, wngy, semg, area



class WatershedEnergy(pl.LightningDataModule):
    def __init__(self, args=None):
        super().__init__()

        self.args = vars(args) if args is not None else {}

        self.fold = self.args.get('fold', FOLD)
        self.image_size = self.args.get('image_size', IMAGE_SIZE)
        self.batch_size = self.args.get('batch_size', BATCH_SIZE)
        self.num_workers = self.args.get('num_workers', NUM_WORKERS)
        self.on_gpu = isinstance(self.args.get('gpu', None), (int, str))

        self.train_ds: WatershedEnergyDataset
        self.valid_ds: WatershedEnergyDataset

    def config(self):
        return None

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

        assert os.path.exists(DIR_IMG), (
            f'Image directory {DIR_IMG} not found.'
            'Load competition data.'
        )

        assert os.path.exists(DIR_SEMSEG), (
            f'Semantic segmentation {DIR_SEMSEG} not found.'
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

        self.train_ds = WatershedEnergyDataset(
            df_train, 
            transform=albu.Compose(aug_tfms(self.image_size))
            )
        self.valid_ds = WatershedEnergyDataset(
            df_valid, 
            transform=albu.Compose(default_tfms(self.image_size))
            )

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
