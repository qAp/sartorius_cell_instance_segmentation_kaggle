
import os, sys
from tqdm.auto import tqdm
import multiprocessing
import numpy as np
import pandas as pd
import cv2

from seggit.data.config import DIR_BASE
from seggit.data.util import annotation_to_semg


train = pd.read_csv(f'{DIR_BASE}/train.csv')

cell_area_list = []
for cell in tqdm(train.itertuples(), total=len(train)):
    semg = annotation_to_semg(cell.annotation)
    cell_area_list.append(semg.sum())
train['cell_area'] = np.array(cell_area_list)

imgids = train['id'].unique()

dir_out = '/kaggle/working/semg'
os.makedirs(dir_out, exist_ok=True)

def _generate_semg(args):
    train, dir_out, imgid = args

    df = train[train['id'] == imgid]
    h_img, w_img = df[['height', 'width']].iloc[0]

    df = df.sort_values('cell_area', axis=0, ascending=False)

    semg = np.zeros((h_img, w_img, 1), dtype=np.float32)
    for cell in df.itertuples():
        semg_cell = annotation_to_semg(cell.annotation,
                                       image_height=h_img, image_width=w_img)
        pxs_cell = semg_cell[:, :, 0].astype(np.bool)
        semg[pxs_cell, :] = 1

    semg = 255 * semg
    semg = semg.astype(np.uint8)
    semg = np.stack([semg[..., 0], semg[..., 0], semg[..., 0]], axis=2)

    cv2.imwrite(f'{dir_out}/{imgid}.png', 
                semg, 
                [cv2.IMWRITE_PNG_COMPRESSION, 9])

    return imgid

args_list = [(train, dir_out, imgid) for imgid in imgids]

p = multiprocessing.Pool(processes=os.cpu_count())
with tqdm(total=len(args_list)) as pbar:
    for imgid in p.imap(_generate_semg, args_list):
        pbar.set_description(f'Processed {imgid}')
        pbar.update(1)
p.close()

