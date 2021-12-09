

import os, sys
import numpy as np
import pandas as pd

import cv2
from skimage.morphology import erosion, dilation, square

from seggit.data.config import DIR_BASE
from seggit.data.util import rle_decode


def generate_semseg(df, width_dilation=2, ):
    '''
    Generate a 2-channel semantic segmentation mask, 

    Args:
        df (pd.DataFrame): Each row for a cell, whose 'annotation' column
            is the run-length encoding for the cell. 'height' and 'width'
            are the height and width of some underlying image.  They
            need to be the same for all rows in `df`.
        width_dilation (int): The width of the square to be used in the
            dilation of a semantic segmentation mask. 
    Returns:
        semseg (np.array, np.bool, (H, W, 2)): Channel 0 is 
            the mask for the cells, where overlapped regions 
            between cells are removed. The larger `width_dilation` is 
            the larger area is removed.  Channel 1 is the area removed 
            due to cells overlapping. 
    '''
    hs = df['height'].unique()
    assert len(hs) == 1
    height = hs[0]

    ws = df['width'].unique()
    assert len(ws) == 1
    width = ws[0]

    semseg = np.zeros((height, width, 2), dtype=np.bool)
    for i, cell in enumerate(df.itertuples()):
        msk = rle_decode(cell.annotation, (height, width, 1))
        msk = msk.astype(np.bool)

        dmsk = dilation(msk[..., 0], square(width_dilation))
        dmask = dilation(semseg[..., 0], square(width_dilation))

        overlap = dmask & dmsk

        semseg[overlap, 1] = True
        semseg[..., 0] = semseg[..., 0] ^ msk[..., 0]
        semseg[overlap, 0] = False

    return semseg




