import numpy as np

__all__ = ['DIR_BASE', 'DIR_KFOLD', 'DIR_IMG', 'DIR_MASK']

DIR_BASE = '/kaggle/input/sartorius-cell-instance-segmentation/'
DIR_KFOLD = '/kaggle/input/sardata-kfold/kfold'
DIR_IMG = f'{DIR_BASE}/train'
DIR_MASK = '/kaggle/input/sardata-train-mask/train_mask'
DIR_AREA = '/kaggle/input/sardata-instance-area/instance_area'
DIR_DTFM = '/kaggle/input/sardata-distance-transform/distance_transform'
DIR_UVEC = '/kaggle/input/sardata-uvec/uvec'
DIR_ENERGY = '/kaggle/input/sardata-watershed-energy/watershed_energy'

# Generated with data.util.define_watershed_energy_bins()
WATERSHED_ENERGY_BINS = np.array(
    [0.955,  1.369,  1.91,  2.324,  2.739,  3.279,  3.694,  4.108,
     4.649,  5.063,  5.477,  6.018, 10., 20., 40., 80.])
