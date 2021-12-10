'''
Compute the competition images' mean and std.
'''

import os, sys
from tqdm.auto import tqdm
import multiprocessing
import numpy as np
import cv2

from seggit.data.config import DIR_IMG


def _compute_mean_std(args):
    fn = args
    img = cv2.imread(f'{DIR_IMG}/{fn}')
    img = img.astype(np.float32)
    return fn, img.mean(), img.std()

args_list = os.listdir(DIR_IMG)

mean_list = []
std_list = []
p = multiprocessing.Pool(processes=os.cpu_count())
with tqdm(total=len(args_list)) as pbar:
    for fn, mean_img, std_img in p.imap(_compute_mean_std, args_list):
        mean_list.append(mean_img)
        std_list.append(std_img)
        pbar.set_description(f'Processed {fn}')
        pbar.update(1)
p.close()

mean_dataset = np.array(mean_list).mean()
std_dataset = np.array(std_list).mean()
print(f'Dataset mean={mean_dataset} std={std_dataset}')


