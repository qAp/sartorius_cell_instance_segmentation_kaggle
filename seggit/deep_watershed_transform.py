
import numpy as np
from seggit.data.util import padto_divisible_by32
import torch

from seggit.data.config import WATERSHED_ENERGY_BINS
from seggit.models import WatershedNet
from seggit.lit_models import WatershedLitModel


def load_watershed_litmodel(checkpoint_path=None):
    model = WatershedNet()
    
    lit_model = WatershedLitModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model)
    
    lit_model.eval()
    return lit_model


class DeepWatershedTransform:
    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                                   else 'cpu')

        self.model = load_watershed_litmodel(checkpoint_path)
        self.model.to(self.device)

    @torch.no_grad()
    def predict_logits(self, img, semg):
        '''
        Args:
            img (np.array[float]): 
                Normalised image. Shape (N, H, W, 3) 
            semg  (np.array[float]): 
                Semantic segmentation. Shape (N, H, W, 1)

        Returns:
            logits (np.array[float]): Logits. Shape (N, H, W, 17)
        '''
        img = torch.from_numpy(img).permute(0, 3, 1, 2)
        semg = torch.from_numpy(semg).permute(0, 3, 1, 2)
        
        img = img.to(self.device)
        semg = semg.to(self.device)
        logits = self.model(img, semg)

        logits = logits.permute(0, 2, 3, 1).data.cpu().numpy()
        return logits

    def predict_logits_tta(self, img, semg):
        '''
        Args:
            img (np.array[float]): 
                Normalised image. Shape (N, H, W, 3) 
            semg  (np.array[float]): 
                Semantic segmentation. Shape (N, H, W, 1)

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
                ss = semg[:,::-1,:,:] if flipud else semg
                mm = np.rot90(mm, k=angle//90, axes=(1, 2))
                ss = np.rot90(ss, k=angle//90, axes=(1, 2))

                ll = self.predict_logits(mm.copy(), ss.copy())

                ll = np.rot90(ll, k=(360 - angle)//90, axes=(1, 2))
                ll = ll[:,::-1,:,:] if flipud else ll

                logits += ll
        
        logits /= (len(do_flipud) * len(rot_angles))
        return logits

    def predict(self, img, semg, tta=False):
        '''
        Args:
            img (np.array[float]): 
                Normalised image. Shape (H, W, 3) 
            semg  (np.array[float]): 
                Semantic segmentation. Shape (H, W, 1)

        Returns:
            wngy (np.array[float]): 
                Discrete watershed energy. Shape (N, H, W, 1)
        '''
        pad_img, img = padto_divisible_by32(img)
        img = img[None, ...]

        pad_semg, semg = padto_divisible_by32(semg)
        semg = semg[None, ...]


        if tta:
            logits = self.predict_logits_tta(img, semg)
        else:
            logits = self.predict_logits(img, semg)

        wngy = logits.argmax(axis=3)[..., None]
        wngy = wngy.astype(np.float32)

        wngy = wngy[:, 
                    pad_img['y0']:-pad_img['y1'],
                    pad_img['x0']:-pad_img['x1'],
                    :]
        return wngy
