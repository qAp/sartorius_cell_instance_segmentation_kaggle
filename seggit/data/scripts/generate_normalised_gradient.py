
import os
import multiprocessing
from tqdm.auto import tqdm
from seggit.data.config import DIR_DTFM
from seggit.data.util import _generate_normalised_gradient

dir_uvec = '/kaggle/working/uvec'

imgids = [fn.split('.')[0] for fn in os.listdir(DIR_DTFM)]
args_list = [(imgid, dir_uvec) for imgid in imgids]

os.makedirs(dir_uvec, exist_ok=True)

p = multiprocessing.Pool(processes=os.cpu_count())

with tqdm(total=len(args_list)) as pbar:
    for imgid in p.imap(_generate_normalised_gradient, args_list):
        pbar.set_description(f'Processing {imgid}')
        pbar.update(1)

p.close()