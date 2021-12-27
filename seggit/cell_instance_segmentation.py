
import os
import argparse
import numpy as np
import torch
import cv2
import skimage.morphology

from seggit.data.config import MEAN_IMAGE, STD_IMAGE
from seggit.data.util import padto_divisible_by32
from seggit.cell_semantic_segmentation import SemanticSegmenter
from seggit.deep_watershed_transform import DeepWatershedTransform



PTH_UNET = 'unet.ckpt'
PTH_WN = 'wn.ckpt'
PP_SEMSEG = 0


def watershed_cut(wngy, semg, threshold=1, selem_width=3):
    '''
    Args:
        wngy (H, W, 1) np.array float
        semg (H, W, 1) np.array float
    Returns:
        cclabels_out (H, W, 1) np.array float: Instance IDs
    '''
    semg = semg.astype(np.bool)
    ccimg = (wngy > threshold) * semg
    ccimg_nosmall = skimage.morphology.remove_small_objects(ccimg, 
                                                            min_size=20)
    ccimg_nohole = skimage.morphology.remove_small_holes(ccimg_nosmall)
    cclabels = skimage.morphology.label(ccimg_nohole)
    
    ccids = np.unique(cclabels)[1:]

    cclabels_out = np.zeros_like(wngy)
    for id in ccids:
        ccimg_id = (cclabels == id)
        ccimg_id_dilated = skimage.morphology.binary_dilation(
            ccimg_id,
            selem=np.ones(3 * (selem_width,)).astype(np.bool)
        )

        cclabels_out[ccimg_id_dilated] = id
        
    return cclabels_out


class CellSegmenter:
    def __init__(self, args=None):
        self.args = vars(args) if args is not None else {}

        self.pth_unet = self.args.get('pth_unet', PTH_UNET)
        self.pth_wn = self.args.get('pth_wn', PTH_WN)
        self.tta_semseg = self.args.get('tta_semseg', False)
        self.tta_wngy = self.args.get('tta_wngy', False)
        self.pp_semseg = self.args.get('pp_semseg', PP_SEMSEG)

        assert os.path.exists(self.pth_unet), f'Cannot find {self.pth_unet}.'
        assert os.path.exists(self.pth_wn), f'Cannot find {self.pth_wn}.'
        print(f'Loading Unet {self.pth_unet}...', end='')
        self.semantic_segmenter = SemanticSegmenter(checkpoint_path=self.pth_unet)
        print('done.')
        print(f'Loading WN {self.pth_wn}...', end='')
        self.dwt = DeepWatershedTransform(checkpoint_path=self.pth_wn)
        print('done.')

    @staticmethod
    def add_argparse_args(parser):
        add = parser.add_argument
        add('--pth_unet', type=str, default=PTH_UNET)
        add('--pth_wn', type=str, default=PTH_WN)
        add('--tta_semseg', action='store_true', default=False)
        add('--tta_wngy', action='store_true', default=False)
        add('--pp_semseg', type=int, default=PP_SEMSEG)

    def pp_semseg0(self, semseg):
        semg = semseg[..., [0]] # + semseg[..., [1]]
        return semg

    def pp_semseg1(self, semseg):
        '''
        semseg: Shape (N, H, W, 3)
        semg: Shape (N, H, W, 1)
        '''
        semg = semseg[..., [0]] + semseg[..., [1]]
        return semg

    def pp_semseg2(self, semseg):
        '''
        semseg: Shape (N, H, W, 3)
        semg: Shape (N, H, W, 1)
        '''
        semg = semseg[..., [0]] + semseg[..., [1]]
        semg = skimage.morphology.binary_dilation(
            semg, selem=np.ones(4 * (4, ))
        )
        semg = semg.astype(np.float32)
        return semg

    def predict(self, pth_img):
        '''
        Args:
            pth_img [str, iter[str]]: Path(s) to image file(s).
        Returns:
            img (np.array[float]): Normalised image. Shape (N, H, W, 3)
            instg  (np.array[float]): 
                Instance segmentation. Shape (N, H, W, 1)
        '''
        img, semseg = self.semantic_segmenter.predict(pth_img, 
                                                      tta=self.tta_semseg)

        if self.pp_semseg == 0:
            semg = self.pp_semseg0(semseg)
        elif self.pp_semseg == 1:
            semg = self.pp_semseg1(semseg)
        elif self.pp_semseg == 2:
            semg = self.pp_semseg2(semseg)

        wngy = self.dwt.predict(img[0,...], semg[0,...], tta=self.tta_wngy) 

        instg = watershed_cut(wngy[0, ...], semg[0, ...])

        instg = instg[None, ...]

        return img, instg


# def main():
#     parser = _setup_parser()
#     args = parser.parser_args()

#     segmenter = CellSegmenter(args)

#     df = pd.read_csv(args.csv)
#     imgids = df['id'].unique()

#     img_pths = [f'{args.dir_img}/{imgid}.png' for imgid in imgids]






         

    



