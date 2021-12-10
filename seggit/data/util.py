

import os, sys
from tqdm.auto import tqdm
import multiprocessing
import numpy as np
import pandas as pd
import cv2
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from seggit.data.config import DIR_BASE, DIR_MASK, DIR_DTFM
from seggit.data.config import WATERSHED_ENERGY_BINS


# Computed with data/scripts/compute_image_mean_std.py
MEAN_IMAGE = 127.96482849121094 
STD_IMAGE = 13.235036849975586

def print_info(a):
    info = (a.dtype, a.shape, a.min(), a.mean(), a.max())
    print(info)


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


def _generate_instance_area(args):
    train, imgid, dir_area = args

    for i, cell in enumerate(train[train['id'] == imgid].itertuples()):
        semseg_cell = rle_decode(mask_rle=cell.annotation,
                                 shape=(cell.height, cell.width, 1),
                                 color=1)

        area_cell = semseg_cell.sum()

        if i == 0:
            semseg = semseg_cell.copy()
            instance_area = area_cell * semseg_cell
        else:
            semseg += semseg_cell
            instance_area += (area_cell * semseg_cell)

    overlap = semseg > 1
    semseg[overlap] = 0
    instance_area[overlap] = 0

    np.save(f'{dir_area}/{imgid}', instance_area)
    return imgid


def _generate_distance_transform(args):
    '''
    Generate distance transform of semantic segmentations.

    Args:
        `dir_dtfm` (str, Path): Directory in which distance transforms
            will be saved in.  One file per image.

    Notes:
    `dtfm` has values between 0 and some positive number. 
    It's not normalised to the range (0, 1).

    '''
    imgid, dir_dtfm = args

    mask = cv2.imread(f'{DIR_MASK}/{imgid}.png')
    mask = mask[..., 0]  # (height, width), (0, 1), np.uint8
    dtfm = cv2.distanceTransform(src=mask, 
                                 distanceType=cv2.DIST_L2, 
                                 maskSize=3)  # (height, width), np.float32

    np.save(f'{dir_dtfm}/{imgid}', dtfm)
    return imgid


def _generate_normalised_gradient(args):
    '''
    Generate normalised gradient of distance transform
    '''
    imgid, dir_uvec = args

    dtfm = np.load(f'{DIR_DTFM}/{imgid}.npy')
    grad_dtfm = np.stack(np.gradient(dtfm), axis=2)
    norm = np.linalg.norm(grad_dtfm, axis=2, keepdims=True)
    uvec = np.nan_to_num(grad_dtfm / norm)

    np.save(f'{dir_uvec}/{imgid}', uvec)
    return imgid
    

def get_most_dtfm_imgid():
    '''
    Get the image id of the image that has the largest 
    number of unique distance transform values.
    '''
    train = pd.read_csv(f'{DIR_BASE}/train.csv')

    imgids = train['id'].unique()

    num_levels = []
    for imgid in tqdm(imgids, total=len(imgids)):
        dtfm = np.load(f'{DIR_DTFM}/{imgid}.npy')
        levels = np.unique(dtfm)
        num_levels.append(len(levels))

    return imgids[np.array(num_levels).argmax()]


def define_watershed_energy_bins():
    '''
    Returns:
        bins (np.array), np.float32: Distance transform values that 
            bounds the watershed energy levels.  `bins[0]` is the 
            lower bound of energy level 1 and the upper bound of
            energy level 0.  `bins[-1]` if the lower bound of
            the highest energy level
    '''
    imgid = get_most_dtfm_imgid()

    n = 20 # Number of lowest distance transform values to use

    dtfm = np.load(f'{DIR_DTFM}/{imgid}.npy')
    bins = np.unique(dtfm)[:n]

    # Manually add several wide bins to cover the higher range.
    bins = np.concatenate([bins, np.array([10, 20, 40, 80])],
                          axis=0)
    assert (np.diff(bins) > 0).all()

    # Left-merge very narrow intervals
    min_bin_width = 0.2
    levels_too_narrow = np.ones_like(bins).astype(np.bool)
    levels_too_narrow[0] = False
    levels_too_narrow[1:] = np.diff(bins) < min_bin_width
    levels_too_narrow[-1] = False
    bins = bins[~levels_too_narrow]

    bins = bins[1:]  # Drop 0, because using np.digitize

    # Pushes bin edges towards cell boundary 
    # and likely ensures all bins are occupied.
    bins = np.round(bins, 3)

    return bins


def _generate_watershed_energy(args):
    '''
    Generate watershed energy by digitizing 
    distance transform using pre-defined energy bins.
    '''
    imgid, dir_energy = args

    dtfm = np.load(f'{DIR_DTFM}/{imgid}.npy')
    energy = np.digitize(dtfm, bins=WATERSHED_ENERGY_BINS)

    np.save(f'{dir_energy}/{imgid}', energy)

    return imgid



