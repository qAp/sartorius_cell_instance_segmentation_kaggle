

import os, sys
import numpy as np
import pandas as pd
import torch
import albumentations as albu
import pytorch_lightning as pl
import cv2
from skimage.morphology import erosion, dilation, square
from seggit.data.config import DIR_BASE, DIR_KFOLD, DIR_IMG, DIR_SEMSEG
from seggit.data.config import MEAN_IMAGE, STD_IMAGE, BAD_SAMPLES
from seggit.data.util import rle_decode 
from seggit.data.transforms import default_tfms, aug_tfms



FOLD = 0
IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 0



class SemSegDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, use_softmax=False):
        super().__init__()
        self.df = df
        self.transform = transform
        self.use_softmax = use_softmax
        self.imgids = [id for id in df['id'].unique() if id not in BAD_SAMPLES]

    def __len__(self):
        return len(self.imgids)

    def __getitem__(self, i):
        imgid = self.imgids[i]

        img = cv2.imread(f'{DIR_IMG}/{imgid}.png')
        semseg = cv2.imread(f'{DIR_SEMSEG}/{imgid}.png')

        if self.transform:
            tfmd = self.transform(image=img, mask=semseg)
            img = tfmd['image']
            semseg = tfmd['mask']

        img = img.astype(np.float32)
        img = (img - MEAN_IMAGE) / (STD_IMAGE)

        if not self.use_softmax:
            semseg = semseg[..., :2]
            
        semseg = np.clip(semseg, 0, 255)
        semseg = (semseg / 255).astype(np.float32)

        return img, semseg
        

class SemSeg(pl.LightningDataModule):
    def __init__(self, args=None):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.fold = self.args.get('fold', FOLD)
        self.image_size = self.args.get('image_size', IMAGE_SIZE)
        self.batch_size = self.args.get('batch_size', BATCH_SIZE)
        self.num_workers = self.args.get('num_workers', NUM_WORKERS)
        self.on_gpu = isinstance(self.args.get('gpus', None), (str, int))

        self.use_softmax = self.args.get('use_softmax', False)
        self.aug = self.args.get('aug', False)

        _tfms = aug_tfms if self.aug else default_tfms
        self.transform = albu.Compose(_tfms(self.image_size))

        self.input_dims = (self.image_size, self.image_size, 3)
        self.output_dims = (self.image_size, self.image_size, 
                            3 if self.use_softmax else 2)

        self.train_ds: SemSegDataset
        self.valid_ds: SemSegDataset

    @staticmethod
    def add_argparse_args(parser):
        add = parser.add_argument
        add('--fold', type=int, default=FOLD)
        add('--image_size', type=int, default=IMAGE_SIZE)
        add('--batch_size', type=int, default=BATCH_SIZE)
        add('--num_workers', type=int, default=NUM_WORKERS)
        add('--use_softmax', action='store_true', default=False)
        add('--aug', action='store_true', default=False)
        return parser

    def config(self):
        return {'input_dims': self.input_dims, 
                'output_dims': self.output_dims}

    def prepare_data(self):
        assert os.path.exists(DIR_KFOLD), (
            f'Kfolds csv files not found at {DIR_KFOLD}. '
            'Generate these with `data.util.generate_kfold` '
            f'and make them available at {DIR_KFOLD}.'
        )

        assert os.path.exists(DIR_IMG), (
            f'Images {DIR_IMG} not found. '
            'Load the competition dataset.'
        )

        assert os.path.exists(DIR_SEMSEG), (
            f'Semantic segmentation masks not found at {DIR_SEMSEG}. '
            'Generate with data/scripts/generate_semseg.py and/or '
            f'make them available at {DIR_SEMSEG}.'
        )

    def setup(self):
        train_df = pd.read_csv(f'{DIR_KFOLD}/train_fold{self.fold}.csv')
        valid_df = pd.read_csv(f'{DIR_KFOLD}/valid_fold{self.fold}.csv')

        self.train_ds = SemSegDataset(train_df, self.transform, self.use_softmax)
        self.valid_ds = SemSegDataset(valid_df, self.transform, self.use_softmax)

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






