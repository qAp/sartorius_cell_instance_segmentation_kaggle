
import os, sys
import multiprocessing
import numpy as np
import pandas as pd


DIR_BASE = '/kaggle/input/sartorius-cell-instance-segmentation/'


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

    write_status = cv2.imwrite(f'train_masks/{imgid}.png', mask)
    return imgid, write_status






