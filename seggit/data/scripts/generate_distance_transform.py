
import os, sys
from tqdm.auto import tqdm
import multiprocessing
from seggit.data.config import DIR_MASK
from seggit.data.util import _generate_distance_transform


dir_dtfm = '/kaggle/working/distance_transform'


imgids = [fn.split('.')[0] for fn in os.listdir(DIR_MASK)]
args_list = [(imgid, dir_dtfm) for imgid in imgids]
os.makedirs(dir_dtfm, exist_ok=True)

p = multiprocessing.Pool(processes=os.cpu_count())
with tqdm(total=len(args_list)) as pbar:
    for imgid in p.imap(_generate_distance_transform, args_list):
        pbar.set_description(f'Processed {imgid}')
        pbar.update(1)
p.close()


