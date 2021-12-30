
import numpy as np
from seggit.data.util import padto_divisible_by32
import torch

from seggit.data.config import WATERSHED_ENERGY_BINS
from seggit.models import WatershedNetBG
from seggit.lit_models import WatershedBGLitModel


def load_watershed_litmodel(checkpoint_path=None):
    model = WatershedNetBG()
    
    lit_model = WatershedBGLitModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model)
    
    lit_model.eval()
    return lit_model


class DeepWatershedBGTransform:
    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                                   else 'cpu')

        self.model = load_watershed_litmodel(checkpoint_path)
        self.model.to(self.device)

    @torch.no_grad()
    def predict_logits(self, img, semseg):
        '''
        Args:
            img (np.array[float]): 
                Normalised image. Shape (N, H, W, 3) 
            semseg  (np.array[float]): 
                Semantic segmentation. Shape (N, H, W, 3)

        Returns:
            logits (np.array[float]): Logits. Shape (N, H, W, 17)
        '''
        img = torch.from_numpy(img).permute(0, 3, 1, 2)
        semseg = torch.from_numpy(semseg).permute(0, 3, 1, 2)
        
        img = img.to(self.device)
        semseg = semseg.to(self.device)
        logits = self.model(img, semseg)

        logits = logits.permute(0, 2, 3, 1).data.cpu().numpy()
        return logits

    def predict_logits_tta(self, img, semseg):
        '''
        Args:
            img (np.array[float]): 
                Normalised image. Shape (N, H, W, 3) 
            semseg  (np.array[float]): 
                Semantic segmentation. Shape (N, H, W, 3)

        Returns:
            logits (np.array[float]): Logits. Shape (N, H, W, 17)
        '''       
        do_flipud = [False, True]
        rot_angles = [0, 90, 180, 270]

        batch_size, height, width, _ = img.shape

        logits = np.zeros(
            (batch_size, height, width, len(WATERSHED_ENERGY_BINS) + 1),
            dtype=np.float32
            )

        for flipud in do_flipud:
            for angle in rot_angles:
                mm = img[:,::-1,:,:] if flipud else img
                ss = semseg[:,::-1,:,:] if flipud else semseg
                mm = np.rot90(mm, k=angle//90, axes=(1, 2))
                ss = np.rot90(ss, k=angle//90, axes=(1, 2))

                ll = self.predict_logits(mm.copy(), ss.copy())

                ll = np.rot90(ll, k=(360 - angle)//90, axes=(1, 2))
                ll = ll[:,::-1,:,:] if flipud else ll

                logits += ll
        
        logits /= (len(do_flipud) * len(rot_angles))
        return logits

    def predict(self, img, semseg, tta=False):
        '''
        Args:
            img (np.array[float]): 
                Normalised image. Shape (H, W, 3) 
            semseg  (np.array[float]): 
                Semantic segmentation. Shape (H, W, 3)

        Returns:
            wngy (np.array[float]): 
                Discrete watershed energy. Shape (N, H, W, 1)
        '''
        pad_img, img = padto_divisible_by32(img, mode='symmetric')
        img = img[None, ...]

        pad_semg, semseg = padto_divisible_by32(semseg, mode='symmetric')
        semseg = semseg[None, ...]

        if tta:
            logits = self.predict_logits_tta(img, semseg)
        else:
            logits = self.predict_logits(img, semseg)

        wngy = logits.argmax(axis=3)[..., None]
        wngy = wngy.astype(np.float32)

        wngy = wngy[:, 
                    pad_img['y0']:-pad_img['y1'],
                    pad_img['x0']:-pad_img['x1'],
                    :]
        return wngy
