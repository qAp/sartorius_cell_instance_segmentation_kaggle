

import os, sys
from tqdm.auto import tqdm
import multiprocessing
import numpy as np
import pandas as pd
import cv2
from skimage.morphology import square, dilation
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from seggit.data.config import DIR_BASE, DIR_KFOLD, DIR_MASK, DIR_DTFM
from seggit.data.config import BAD_SAMPLES
from seggit.data.config import WATERSHED_ENERGY_BINS




def print_info(a):
    info = (a.dtype, a.shape, a.min(), a.mean(), a.max())
    print(info)


def generate_kfold(n_folds=3, dir_kfold='/kaggle/working/kfold'):

    train = pd.read_csv(f'{DIR_BASE}/train.csv')

    df = (
        train[['id', 'cell_type', 'plate_time', 'sample_date', 'sample_id']]
        .groupby('id')
        .first()
    )
    df['num_cells'] = train.groupby('id')['annotation'].nunique()
    df.drop(BAD_SAMPLES, axis=0, inplace=True)
    df.reset_index(inplace=True)

    kf = MultilabelStratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=100)

    X = df[['id']]
    y = df[
        df.columns[df.columns != 'id']
    ]

    train_indices_list = []
    valid_indices_list = []
    for train_indices, valid_indices in kf.split(X.values, y.values):
        train_indices_list.append(train_indices)
        valid_indices_list.append(valid_indices)

    os.makedirs(dir_kfold, exist_ok=True)

    for i in range(n_folds):
        train_df = df.loc[train_indices_list[i]]
        valid_df = df.loc[valid_indices_list[i]]

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


def annotation_to_semg(annotation, image_height=520, image_width=704):
    '''
    Convert RLE to semantic segmentation

    Args:
        annotation (str): Run-length encoding string 
            from train.csv

    Returns:
        semg (np.array, np.float32): Semantic segmentation
            Shape (H, W, 1). Values 0, or 1. 
    '''
    semg = rle_decode(annotation, (image_height, image_width, 1))

    return semg


def semg_to_dtfm(semg):
    '''
    Convert semantic segmentation to distance transform.

    Args:
        semg (np.array, np.float32): Semantic segmentation.
            Shape (H, W, 1). Values 0, or 1.
    
    Returns:
        dtfm (np.array, np.float32): Distance transform.
            Shape (H, W, 1). 
    '''
    dtfm = cv2.distanceTransform(semg.astype(np.uint8),
                                 distanceType=cv2.DIST_L2,
                                 maskSize=3)
    dtfm = dtfm[..., None]

    return dtfm


def dtfm_to_uvec(dtfm):
    '''
    Distance transform to normalised gradient distance transform

    Args:
         dtfm (np.array, np.float32): Distance transform.
            Shape (H, W, 1). 
    Returns:
        uvec (np.array, np.float32): uvec
            Shape (H, W, 2).
    '''
    grad_dtfm = np.stack(np.gradient(dtfm[..., 0]), axis=2)
    norm = np.linalg.norm(grad_dtfm, axis=2, keepdims=True)
    uvec = np.nan_to_num(grad_dtfm / norm)

    return uvec


def dtfm_to_wngy(dtfm):
    '''
    Distance transform to watershed energy.

    Args:
        dtfm (np.array, np.float32): Distance transform.
            Shape (H, W, 1). 
    Returns:
        wngy (np.array, np.int64): Watershed energy levels map.
            Shape (H, W, 1).
    '''
    wngy = np.digitize(dtfm, bins=WATERSHED_ENERGY_BINS)

    return wngy


def get_semg_multicell(df, image_height=520, image_width=704, square_width=5):
    '''
    Semantic segmentation with overlap border
    '''
    df = df.sort_values('cell_area', axis=0, ascending=False)

    mass = np.zeros((image_height, image_width, 1), dtype=np.bool)
    border = np.zeros((image_height, image_width, 1), dtype=np.bool)

    for r in df.itertuples():

        cell = annotation_to_semg(r.annotation, image_height, image_width)
        cell = cell.astype(np.bool)

        celll = dilation(cell[..., 0], selem=square(square_width))
        celll = celll[..., None]

        wall = celll ^ cell

        inter_mw = mass & wall
        inter_bw = border & wall
        border[celll] = False
        border = border | inter_mw | inter_bw

        union_mc = mass | celll
        mass = union_mc ^ wall

    mass = mass.astype(np.float32)
    border = border.astype(np.float32)

    return mass, border


def generate_instance_area(df, image_height=520, image_width=704):
    '''
    Return each pixel's instance area.
    '''
    df = df.sort_values('cell_area', axis=0, ascending=False)
    
    area = np.zeros((image_height, image_width, 1), dtype=np.float32)

    for r in df.itertuples():
        cell = annotation_to_semg(r.annotation, image_height, image_width)
        cell = cell.astype(np.bool)

        area[cell] = cell.sum()

    return area


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
    mask = mask[..., 0]  # (height, width), {0, 1}, np.uint8
    dtfm = cv2.distanceTransform(src=mask, 
                                 distanceType=cv2.DIST_L2, 
                                 maskSize=3)  # (height, width), np.float32

    np.save(f'{dir_dtfm}/{imgid}', dtfm)
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


def find_samples_overlap_gt_cell():
    '''
    Find the samples whose overlap area is greater than
    cell area.  These look like the cells have been
    annotated twice.  The overlap area between successive
    annotations is obviously large.  Wouldn't make sense 
    to use these for training.
    '''
    overlap_large_list = []

    for fn in tqdm(os.listdir(DIR_SEMSEG)):
        semseg = cv2.imread(f'{DIR_SEMSEG}/{fn}')
        semseg = semseg.astype(np.float32)
        semseg = semseg / 255
        semseg = semseg == 1
        area_cell = semseg[..., 0].sum()
        area_overlap = semseg[..., 1].sum()
        if area_overlap > area_cell:
            overlap_large_list.append(fn)

    return overlap_large_list


def padto_divisible_by32(a):
    '''
    Pad to model-compatible shape. i.e. sides are divisible by 32.
    
    Args:
        a (H, W, C) np.array: Array
    '''
    y0, y1 = 16, 16
    x0, x1 = 16, 16

    if a.shape[0] % 32 != 0:
        dy = 32 - a.shape[0] % 32
        y0 += int(dy / 2)
        y1 += dy - int(dy / 2)

    if a.shape[1] % 32 != 0:
        dx = 32 - a.shape[1] % 32
        x0 += int(dx / 2)
        x1 += dx - int(dx / 2)

    a = np.pad(a, 
               pad_width=((y0, y1), (x0, x1), (0, 0)),
               mode='constant')

    assert a.shape[0] % 32 == 0
    assert a.shape[1] % 32 == 0
    
    padding = {'y0': y0, 'y1': y1, 'x0': x0, 'x1': x1}
    return padding, a


