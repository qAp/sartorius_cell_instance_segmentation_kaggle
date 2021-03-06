
import os, sys
from tqdm.auto import tqdm
import argparse
import multiprocessing
import numpy as np
import cv2

from seggit.cell_semantic_segmentation import SemanticSegmenter
from seggit.data.config import DIR_IMG


CHECKPOINT_PATH = 'best.pth'
DIR_OUT = '/kaggle/working/semseg'

parser = argparse.ArgumentParser()
add = parser.add_argument
add('--checkpoint_path', type=str, default=CHECKPOINT_PATH)
add('--dir_img', type=str, default=DIR_IMG)
add('--dir_out', type=str, default=DIR_OUT)
args = parser.parse_args()

os.makedirs(args.dir_out, exist_ok=True)
imgids = [n.split('.')[0] for n in os.listdir(args.dir_img)]
print(f'Number of images to process', len(imgids))

print(f'Loading semantic segmentation model {args.checkpoint_path}...', end='')
segmenter = SemanticSegmenter(checkpoint_path=args.checkpoint_path)
print('done.')

args_list = [(args.dir_img, args.dir_out, imgid) for imgid in imgids]

def _generate_semseg(args):
    dir_img, dir_out, imgid = args
    pth_img = f'{dir_img}/{imgid}.png'

    semseg = segmenter.predict(pth_img)

    semseg = semseg[0, ...]
    semseg = 255 * semseg
    semseg = semseg.astype(np.uint8)

    cv2.imwrite(f'{dir_out}/{imgid}.png', 
              semseg, 
              [cv2.IMWRITE_PNG_COMPRESSION, 9])

    return imgid


with tqdm(total=len(args_list)) as pbar:
    for args in args_list:
        imgid = _generate_semseg(args)
        pbar.set_description(f'Processed {imgid}')
        pbar.update(1)        

# p = multiprocessing.Pool(processes=os.cpu_count())
# with tqdm(total=len(args_list)) as pbar:
#     for imgid in p.imap(_generate_semseg, args_list):
#         pbar.set_description(f'Processed {imgid}')
#         pbar.update(1)
# p.close()
