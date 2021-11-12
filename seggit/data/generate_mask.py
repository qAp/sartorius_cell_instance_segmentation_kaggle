import os
import multiprocessing
import pandas as pd
from tqdm.auto import tqdm

from seggit.data.util import DIR_BASE, _generate_mask

dir_mask = '/kaggle/working/train_mask'
os.makedirs(dir_mask, exist_ok=True)

train = pd.read_csv(f'{DIR_BASE}/train.csv')

args_list = [(train, imgid, dir_mask) for imgid in train['id'].unique()]

p = multiprocessing.Pool(processes=os.cpu_count())
with tqdm(total=len(args_list)) as pbar:
    for imgid, write_status in p.map(_generate_mask, args_list):
        pbar.set_description(f'{imgid}. Write OK: {write_status}')
p.close()
