
import os, sys
from tqdm.auto import tqdm
import multiprocessing
import numpy as np
import pandas as pd
import cv2

from seggit.data.config import DIR_BASE
from seggit.data.semseg import generate_semseg

width_dilation = 2

train = pd.read_csv(f'{DIR_BASE}/train.csv')
imgids = train['id'].unique()

dir_out = f'/kaggle/working/semseg_wdilate{width_dilation}'
os.makedirs(dir_out, exist_ok=True)


def _generate_semseg_mask(args):
    train, dir_out, imgid = args
    df_img = train[train['id'] == imgid]
    semseg = generate_semseg(df_img, width_dilation)

    semseg = 255 * semseg
    semseg = semseg.astype(np.uint8)
    ch2 = np.zeros_like(semseg[..., 0])
    semseg = np.stack(
        [
            semseg[..., 0], semseg[..., 1], ch2
            ], 
            axis=2)

    cv2.imwrite(f'{dir_out}/{imgid}.png', 
                semseg, 
                [cv2.IMWRITE_PNG_COMPRESSION, 9])

    return imgid


args_list = [(train, dir_out, imgid) for imgid in imgids]

p = multiprocessing.Pool(processes=os.cpu_count())
with tqdm(total=len(args_list)) as pbar:
    for imgid in p.imap(_generate_semseg_mask, args_list):
        pbar.set_description(f'Processed {imgid}')
        pbar.update(1)
p.close()

