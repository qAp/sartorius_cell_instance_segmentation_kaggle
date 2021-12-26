
import os
import numpy as np
import pandas as pd
import cv2
import skimage.morphology
import torch
import albumentations as albu
import pytorch_lightning as pl
from seggit.data.config import (DIR_KFOLD, DIR_IMG, 
                                DIR_SEMSEG, DIR_SEMSEG_MODEL, 
                                DIR_AREA,
                                MEAN_IMAGE, STD_IMAGE)
from seggit.data.util import semg_to_dtfm, dtfm_to_uvec
from seggit.data.transforms import default_tfms, aug_tfms


FOLD = 0
IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 0
SELEM_WIDTH = None



class InstanceDirectionRealDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, selem_width=None):
        super().__init__()

        self.df = df
        self.transform = transform
        self.selem_width = selem_width

        self.imgids = df['id'].unique()

    def __len__(self):
        return len(self.imgids)

    def __getitem__(self, i):
        imgid = self.imgids[i]

        img = cv2.imread(f'{DIR_IMG}/{imgid}.png')
        semseg = cv2.imread(f'{DIR_SEMSEG}/{imgid}.png')
        semsegm = cv2.imread(f'{DIR_SEMSEG_MODEL}/{imgid}.png')
        area = np.load(f'{DIR_AREA}/{imgid}.npy')

        semseg = semseg.astype(np.float32)
        semsegm = semsegm.astype(np.float32)
 
        if self.transform:
            mask = np.concatenate([semseg, semsegm, area], axis=2)
            tfmd = self.transform(image=img, mask=mask)
            img = tfmd['image']
            mask = tfmd['mask']
            semseg = mask[..., :3]
            semsegm = mask[..., 3:6]
            area = mask[..., [6]]

        img = (img - MEAN_IMAGE) / STD_IMAGE
        img = img.astype(np.float32)

        # G.t uvec needs to be from instance-resolved ss 
        semg = semseg[..., [0]] / 255
        dtfm = semg_to_dtfm(semg)
        uvec = dtfm_to_uvec(dtfm)

        # For model and loss input, use instance-unresolved ss,
        # as will be expected from the Unet during inference. 
        semg = (semsegm[..., [0]] + semsegm[..., [1]]) / 255
        if self.selem_width is not None:
            semg = skimage.morphology.binary_dilation(
                semg, 
                selem=np.ones(3 * (self.selem_width, ))
                )

        return img, uvec, semg, area


class InstanceDirectionReal(pl.LightningDataModule):
    def __init__(self, args=None):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.fold = self.args.get('fold', FOLD)
        self.image_size = self.args.get('image_size', IMAGE_SIZE)
        self.batch_size = self.args.get('batch_size', BATCH_SIZE)
        self.num_workers = self.args.get('num_workers', NUM_WORKERS)
        self.on_gpu = isinstance(self.args.get('gpus', None) , (int, str))

        self.train_transform = albu.Compose(aug_tfms(self.image_size))
        self.valid_transform = albu.Compose(default_tfms(self.image_size))
        self.selem_width = self.args.get('selem_width', SELEM_WIDTH)

        self.train_ds: InstanceDirectionRealDataset
        self.valid_ds: InstanceDirectionRealDataset

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--fold', type=int, default=FOLD)
        parser.add_argument('--image_size', type=int, default=IMAGE_SIZE)
        parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
        parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)
        parser.add_argument('--selem_width', type=int, default=SELEM_WIDTH)

    def config(self):
        return None

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
            'data/scripts/make_semseg_target.py'
        )

        assert os.path.exists(DIR_SEMSEG_MODEL), (
            f'Semantic segmentation {DIR_SEMSEG_MODEL} not found.'
            'Load this or generate with ' 
            'cell_semantic_segmentation.py'
        )        

        assert os.path.exists(DIR_AREA), (
            f'Instance area {DIR_AREA} not found.'
            'Load this or generate with'
            'data/scripts/generate_instance_area.py'
        )

    def setup(self):
        try:
            train_df = pd.read_csv(f'{DIR_KFOLD}/train_fold{self.fold}.csv')
            valid_df = pd.read_csv(f'{DIR_KFOLD}/valid_fold{self.fold}.csv')
        except FileNotFoundError:
            print(f'Fold csv files not found at {DIR_KFOLD}')

        self.train_ds = InstanceDirectionRealDataset(
            df=train_df, 
            transform=self.train_transform,
            selem_width=self.selem_width)

        self.valid_ds = InstanceDirectionRealDataset(
            df=valid_df, 
            transform=self.valid_transform,
            selem_width=self.selem_width)

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
