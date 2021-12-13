
import os, sys
from tqdm.auto import tqdm
import multiprocessing
import numpy as np
import pandas as pd
import cv2

from seggit.data.config import DIR_BASE
from seggit.data.util import annotation_to_semg, get_semg_multicell


train = pd.read_csv(f'{DIR_BASE}/train.csv')

print('Calculating cell areas...')
cell_area_list = []
for cell in tqdm(train.itertuples(), total=len(train)):
    semg = annotation_to_semg(cell.annotation)
    cell_area_list.append(semg.sum())
train['cell_area'] = np.array(cell_area_list)

imgids = train['id'].unique()

dir_out = '/kaggle/working/semg'
os.makedirs(dir_out, exist_ok=True)

args_list = [(train, dir_out, imgid) for imgid in imgids]

def _generate_semg(args):
    train, dir_out, imgid = args

    df = train[train['id'] == imgid]
    image_height, image_width = df[['height', 'width']].iloc[0]

    mass, border = get_semg_multicell(df, image_height, image_width, 
                                      square_width=3)

    ch2 = np.zeros_like(mass)
    semg = np.concatenate([mass, border, ch2], axis=2)
    semg = 255 * semg
    semg = semg.astype(np.uint8)
    semg[..., 2] = 255 - semg[..., 0] - semg[..., 1]

    cv2.imwrite(f'{dir_out}/{imgid}.png', semg, 
                [cv2.IMWRITE_PNG_COMPRESSION, 9])

    return imgid


print('Generate semantic segmentation targets...')
p = multiprocessing.Pool(processes=os.cpu_count())
with tqdm(total=len(args_list)) as pbar:
    for imgid in p.imap(_generate_semg, args_list):
        pbar.set_description(f'Processed {imgid}')
        pbar.update(1)
p.close()

print('finished.')

