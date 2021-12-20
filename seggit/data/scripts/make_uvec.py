
import os
import multiprocessing
from tqdm.auto import tqdm
import numpy as np
import cv2

from seggit.data.config import DIR_SEMSEG
from seggit.data.util import semg_to_dtfm, dtfm_to_uvec



dir_uvec = '/kaggle/working/uvec'
os.makedirs(dir_uvec, exist_ok=True)

imgids = [fn.split('.')[0] for fn in os.listdir(DIR_SEMSEG)]
args_list = [(imgid, dir_uvec) for imgid in imgids]

def _generate_normalised_gradient(args):
    imgid, dir_uvec = args
    semseg = cv2.imread(f'{DIR_SEMSEG}/{imgid}.png')
    semg = semseg[..., [0]]
    semg = semg.astype(np.float32)
    semg = semg / 255

    dtfm = semg_to_dtfm(semg)
    uvec = dtfm_to_uvec(dtfm)

    np.save(f'{dir_uvec}/{imgid}', uvec)
    return imgid


p = multiprocessing.Pool(processes=os.cpu_count())
with tqdm(total=len(args_list)) as pbar:
    for imgid in p.imap(_generate_normalised_gradient, args_list):
        pbar.set_description(f'Processing {imgid}')
        pbar.update(1)
p.close()