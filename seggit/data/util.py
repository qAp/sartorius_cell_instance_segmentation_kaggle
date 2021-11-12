

import os, sys
import multiprocessing
import numpy as np
import pandas as pd
import cv2
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

__all__ = ['DIR_BASE', 'DIR_KFOLD']

DIR_BASE = '/kaggle/input/sartorius-cell-instance-segmentation/'
DIR_KFOLD = '/kaggle/input/sardata-kfold/kfold'


def generate_kfold(dir_kfold='/kaggle/working/kfold'):

    df = pd.read_csv(f'{DIR_BASE}/train.csv')
    X = df.values
    y = df[['cell_type', 'plate_time', 'sample_date', 'sample_id']].values

    n_splits = 5
    kf = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=100)

    train_indices_list = []
    valid_indices_list = []
    for train_indices, valid_indices in kf.split(X, y):
        train_indices_list.append(train_indices)
        valid_indices_list.append(valid_indices)

    os.makedirs(dir_kfold, exist_ok=True)
    for i in range(n_splits):
        train_df = df.loc[train_indices_list[i]]
        valid_df = df.loc[valid_indices_list[i]]

        if i == 0:
            print('Train')
            print(train_df['cell_type'].value_counts())
            print('Valid')
            print(valid_df['cell_type'].value_counts())

        train_df.to_csv(f'{dir_kfold}/train_fold{i}.csv', index=False)
        valid_df.to_csv(f'{dir_kfold}/valid_fold{i}.csv', index=False)

    print(
        f'Folds generated and saved in {dir_kfold}. Move them to {DIR_KFOLD}')


def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height, width, channels) of array to return 
    color: color for the mask
    Returns numpy array (mask)
    
    ref: https://www.kaggle.com/inversion/run-length-decoding-quick-start    
    '''
    s = mask_rle.split()

    starts = list(map(lambda x: int(x) - 1, s[0::2]))
    lengths = list(map(int, s[1::2]))
    ends = [x + y for x, y in zip(starts, lengths)]

    img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)

    for start, end in zip(starts, ends):
        img[start: end] = color

    return img.reshape(shape)


def _generate_mask(args):
    train, imgid, dir_mask = args

    img = cv2.imread(f'{DIR_BASE}/train/{imgid}.png')

    mask_shape = img.shape[:2] + (1,)
    mask = np.zeros(mask_shape, dtype=np.float32)
    for cell in train[train['id'] == imgid].itertuples():
        mask = mask + rle_decode(cell.annotation, mask_shape)

    mask[mask > 1] = 0

    write_status = cv2.imwrite(f'{dir_mask}/{imgid}.png', mask)
    return imgid, write_status

