
import os, sys
import argparse
import numpy as np
import torch
import cv2

from seggit.data.util import padto_divisible_by32
from seggit.data.config import MEAN_IMAGE, STD_IMAGE
from seggit.data import SemSeg
from seggit.models.util import create_segmentation_model
from seggit.lit_models import SemSegLitModel



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


class SemanticSegmenter:

    def __init__(self, checkpoint_path=None, tta=False):
        self.checkpoint_path = checkpoint_path
        self.tta = tta
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')        

        self.model = load_semseg_litmodel(checkpoint_path)
        self.model.to(self.device)

    @torch.no_grad()
    def predict_logits(self, img):
        '''
        Args:
            img (N, H, W, 3) np.array: Normalised image.
        Returns:
            logits (N, H, W, 3) np.array float: Model logits. 
                Channel 0 == cell.  
                Channel 1 == overlap wall. 
                Channel 2 == everything else.
        '''
        img = torch.from_numpy(img).permute(0, 3, 1, 2)
        img = img.to(self.device)
        logits = self.model(img)
        logits = logits.permute(0, 2, 3, 1).data.cpu().numpy()
        return logits

    def predict_logits_tta(self, img):
        '''
        Args:
            img (N, H, W, 3) np.array: Normalised image.
        Returns:
            logits (N, H, W, 3) np.array float: Model logits.            
        '''
        do_flipud = [True, False]
        rot_angles = [0, 90, 180, 270]

        batch_size, height, width, _ = img.shape
        
        logits = np.zeros((batch_size, height, width, 3), np.float32)
        for j, flipud in enumerate(do_flipud):
            for i, angle in enumerate(rot_angles):
                mm = img[:,::-1,:,:] if flipud else img
                mm = np.rot90(mm, k=angle//90, axes=(1, 2))
                
                ll = self.predict_logits(mm)

                ll = np.rot90(ll, k=(360 - angle)//90, axes=(1, 2))
                ll = ll[:,::-1,:,:] if flipud else ll

                logits += ll

        logits /= (len(do_flipud) * len(rot_angles))
        return logits

    @torch.no_grad()
    def predict(self, pth_img):
        '''
        Args:
            pth_img [str, iter[str]]: Path(s) to image file(s).
        Returns:
            semseg (N, H, W, 3) np.array: Semantic segmentation
        '''
        # sample
        img = cv2.imread(pth_img)
        pad_img, img = padto_divisible_by32(img)
        img = img.astype(np.float32)
        img = (img - MEAN_IMAGE) / STD_IMAGE
        img = img[None, ...] 
 
        if self.tta:
            logits = self.predict_logits_tta(img)
        else:
            logits = self.predict_logits(img)

        semseg = logits.argmax(axis=3)[..., None]
        semseg = np.concatenate([semseg==0, semseg==1, semseg==2], axis=3)
        semseg = semseg.astype(np.float32)

        semseg = semseg[:, 
                        pad_img['y0']: -pad_img['y1'],
                        pad_img['x0']: -pad_img['x1'], 
                        :]

        return semseg


