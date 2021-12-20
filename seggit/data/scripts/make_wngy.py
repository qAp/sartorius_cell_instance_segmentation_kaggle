
import os, sys
import numpy as np
import cv2
from tqdm.auto import tqdm
import multiprocessing
from seggit.data.util import dtfm_to_wngy, semg_to_dtfm
from seggit.data.config import DIR_SEMSEG, WATERSHED_ENERGY_BINS


dir_energy = '/kaggle/working/wngy'
os.makedirs(dir_energy, exist_ok=True)

imgids = [f.split('.')[0] for f in os.listdir(DIR_SEMSEG)]
args_list = [(imgid, dir_energy) for imgid in imgids]

def _generate_watershed_energy(args):
    imgid, dir_energy = args

    semseg = cv2.imread(f'{DIR_SEMSEG}/{imgid}.png')
    semg = semseg[..., [0]]
    semg = semg.astype(np.float32)
    semg = semg / 255    

    dtfm = semg_to_dtfm(semg)
    wngy = dtfm_to_wngy(dtfm)

    np.save(f'{dir_energy}/{wngy}')

    return imgid


p = multiprocessing.Pool(processes=os.cpu_count())
with tqdm(total=len(args_list)) as pbar:
    for imgid in p.imap(_generate_watershed_energy, args_list):
        pbar.set_description(f'Processing {imgid}...')
        pbar.update(1)
p.close()
