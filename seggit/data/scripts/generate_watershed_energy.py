
import os, sys
from tqdm.auto import tqdm
import multiprocessing
from seggit.data.util import _generate_watershed_energy
from seggit.data.config import DIR_DTFM, WATERSHED_ENERGY_BINS


dir_energy = '/kaggle/working/watershed_energy'
os.makedirs(dir_energy, exist_ok=True)

imgids = [f.split('.')[0] for f in os.listdir(DIR_DTFM)]
args_list = [(imgid, dir_energy) for imgid in imgids]

p = multiprocessing.Pool(processes=os.cpu_count())

with tqdm(total=len(args_list)) as pbar:
    for imgid in p.imap(_generate_watershed_energy, args_list):
        pbar.set_description(f'Processing {imgid}...')
        pbar.update(1)

p.close()
