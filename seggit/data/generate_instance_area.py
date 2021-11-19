
import os
import multiprocessing
from tqdm.auto import tqdm
import pandas as pd
from seggit.data.config import DIR_BASE
from seggit.data.util import _generate_instance_area


dir_area = '/kaggle/working/instance_area'
os.makedirs(dir_area, exist_ok=True)

train = pd.read_csv(f'{DIR_BASE}/train.csv')
imgids = train['id'].unique()

args_list = [(train, imgid, dir_area) for imgid in imgids]

p = multiprocessing.Pool(processes=os.cpu_count())

with tqdm(total=len(args_list)) as pbar:
    for imgid in p.imap(_generate_instance_area, args_list):
        pbar.set_description(f'Processed {imgid}')
        pbar.update(1)

p.close()


