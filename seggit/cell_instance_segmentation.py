
import os
import argparse
import numpy as np
import torch
import cv2
import skimage.morphology

from seggit.data.config import MEAN_IMAGE, STD_IMAGE
from seggit.data.util import padto_divisible_by32
from seggit.data import SemSeg
from seggit.models.util import create_segmentation_model
from seggit.models import WatershedNet
from seggit.lit_models import SemSegLitModel, WatershedLitModel



PTH_UNET = 'unet.ckpt'
PTH_WN = 'wn.ckpt'


def load_semseg_litmodel(checkpoint_path=None):
    parser = argparse.ArgumentParser()
    SemSeg.add_argparse_args(parser)
    SemSegLitModel.add_argparse_args(parser)
    args = parser.parse_args([
        '--use_softmax', 
        '--encoder_name', 'resnet152',
    ])
    
    data = SemSeg(args)
    model = create_segmentation_model(data.config(), args)
    
    lit_model = SemSegLitModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path, 
        model=model)
    
    lit_model.eval()
    return lit_model


def load_watershed_litmodel(checkpoint_path=None):
    model = WatershedNet()
    
    lit_model = WatershedLitModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model)
    
    lit_model.eval()
    return lit_model


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
        assert os.path.exists(self.pth_unet), f'Cannot find {self.pth_unet}.'
        assert os.path.exists(self.pth_wn), f'Cannot find {self.pth_wn}.'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f'Loading Unet {self.pth_unet}...', end='')
        self.unet = load_semseg_litmodel(checkpoint_path=self.pth_unet)
        self.unet.to(self.device)
        print('done.')

        print(f'Loading WN {self.pth_wn}...', end='')
        self.wn = load_watershed_litmodel(checkpoint_path=self.pth_wn)
        self.wn.to(self.device)
        print('done.')

    @staticmethod
    def add_argparse_args(parser):
        add = parser.add_argument
        add('--pth_unet', type=str, default=PTH_UNET)
        add('--pth_wn', type=str, default=PTH_WN)

    def pp0(self, semseg):
        semg = semseg[..., [0]] # + semseg[..., [1]]
        return semg

    def pp1(self, pr):
        semg_pred = (pr == 0) + (pr == 1)
        semg_pred = skimage.morphology.binary_dilation(
            semg_pred, 
            selem=np.ones(3 * (4, ))
            )
        semg_pred = semg_pred.astype(np.float32)
        return semg_pred 

    @torch.no_grad()
    def predict_semseg(self, img):
        '''
        Args:
            img (N, H, W, 3) np.array
        Returns:
            semseg (N, H, W, 3) np.array
        '''
        img = torch.from_numpy(img).permute(0, 3, 1, 2)

        img = img.to(self.device)
        logits = self.unet(img)
        
        semseg = logits.argmax(dim=1, keepdim=True)
        semseg = torch.cat([semseg==0, semseg==1, semseg==2], dim=1)
        semseg = semseg.type(torch.float32)    
        semseg = semseg.permute(0, 2, 3, 1).detach().cpu().numpy()
        return semseg

    @torch.no_grad()
    def predict_wngy(self, img, semg):
        '''
        Args:
            img (N, H, W, 3) np.array
            semg (N, H, W, 1) np.array

        Returns:
            wngy (N, H, W, 1) np.array
        '''
        img = torch.from_numpy(img).permute(0, 3, 1, 2)
        semg = torch.from_numpy(semg).permute(0, 3, 1, 2)
        
        img = img.to(self.device)
        semg = semg.to(self.device)
        logits = self.wn(img, semg)
        
        wngy = logits.argmax(dim=1, keepdim=True)
        wngy = wngy.type(torch.float32)
        
        wngy = wngy.permute(0, 2, 3, 1).detach().cpu().numpy()
        return wngy

    @torch.no_grad()
    def predict(self, pth_img):
        '''
        Args:
            pth_img [str, iter[str]]: Path(s) to image file(s).
        Returns:
            instg (N, H, W, 1) np.array: Instance segmentation
        '''
        # sample
        img = cv2.imread(pth_img)
        pad_img, img = padto_divisible_by32(img)
        img = img.astype(np.float32)
        img = (img - MEAN_IMAGE) / STD_IMAGE

        # batch
        img = img[None, ...]
        semseg = self.predict_semseg(img)
        semg = self.pp0(semseg) 
        wngy = self.predict_wngy(img, semg) 

        # sample
        instg = watershed_cut(wngy[0, ...], semg[0, ...])

        # batch
        instg = instg[None, 
                      pad_img['y0']: -pad_img['y1'], 
                      pad_img['x0']: -pad_img['x1'], 
                      :]

        return instg


# def main():
#     parser = _setup_parser()
#     args = parser.parser_args()

#     segmenter = CellSegmenter(args)

#     df = pd.read_csv(args.csv)
#     imgids = df['id'].unique()

#     img_pths = [f'{args.dir_img}/{imgid}.png' for imgid in imgids]






         

    



