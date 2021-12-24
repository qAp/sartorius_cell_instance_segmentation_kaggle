
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
    ccimg_nosmall = skimage.morphology.remove_small_objects(ccimg, min_size=20,)
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

        self.pth_unet = self.args.get('pth_unet')
        self.pth_wn = self.args.get('pth_wn')

    def load_models(self):
        self.unet = load_semseg_litmodel(checkpoint_path=self.pth_unet)
        self.wn = load_watershed_litmodel(checkpoint_path=self.pth_wn)

    @torch.no_grad()
    def predict_semseg(img):
        '''
        Args:
            img (N, H, W, 3) np.array
        Returns:
            semseg (N, H, W, 3) np.array
        '''
        img = torch.from_numpy(img).permute(0, 3, 1, 2)
        
        logits = unet_litmodel(img)
        
        semseg = logits.argmax(dim=1, keepdim=True)
        semseg = torch.cat([semseg==0, semseg==1, semseg==2], dim=1)
        semseg = semseg.type(torch.float32)    
        semseg = semseg.permute(0, 2, 3, 1).numpy()
        return semseg

    @torch.no_grad()
    def predict_wngy(img, semg):
        '''
        Args:
            img (N, H, W, 3) np.array
            semg (N, H, W, 1) np.array
        '''
        img = torch.from_numpy(img).permute(0, 3, 1, 2)
        semg = torch.from_numpy(semg).permute(0, 3, 1, 2)
        
        logits = wn_litmodel(img, semg)
        
        wngy = logits.argmax(dim=1, keepdim=True)
        wngy = wngy.type(torch.float32)
        
        wngy = wngy.permute(0, 2, 3, 1).numpy()
        return wngy

    



