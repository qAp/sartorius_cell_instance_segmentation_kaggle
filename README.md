
# A solution to the Sartorius Cell Instance Segmentation Kaggle
https://www.kaggle.com/c/sartorius-cell-instance-segmentation


## Solution summary
1. Semantic segmentation by Unet.
2. Instance segmentation by further processing of semantic segmentation with [Deep Watershed Transform](https://arxiv.org/pdf/1611.08303.pdf).

Deep Watershed Transform Network:  
<img src="images/direction_net.png" width=900 height=350>

## Semantic segmentation (Unet)
1. To generate training target:
```
python seggit/data/scripts/make_semseg_target.py
```
2. To make a training run:
```
python seggit/training/run_segmentation.py
```
3. To make inference:
```
from seggit.cell_semantic_segmentation import SemanticSegmenter
segmenter = SemanticSegmenter(checkpoint_path='best.pth')
img, semseg = segmenter.predict('sample.png')
```

## Direction Net (DN)
1. To generate training target:
```
python seggit/data/scripts/make_uvec.py
```
2. To make a training run:
```
python training/run_direction.py
```

## Watershed Transform Net (WTN)
1. To generate training target:
```
python seggit/data/scripts/make_wngy.py
```
2. To make a training run:
```
python training/run_energy.py
```

## Deep Watershed Transform end-to-end (WN)
1. To make a training run:
```
python training/run_watershed.py
```
2. To make an inference:
```
from seggit.deep_watershed_transform import DeepWatershedTransform

dwt = DeepWatershedTransform(checkpoint_path='best.pth')
wngy = dwt.predict(img, semg)
```

## Cell instance segmentation (Unet + WN)
To make an inference :
```
from seggit.cell_instance_segmentation import CellSegmenter

parser = argparse.ArgumentParser()
CellSegmenter.add_argparse_args(parser)
args = parser.parse_args()
args.pth_unet = 'best_unet.pth'
args.pth_wn = 'best_wn.pth'

segmenter = CellSegmenter(args)

img, instg = segmenter.predict('sample.png')
```




# References
- [[ods.ai] topcoders, 1st place solution](https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741)
- [@hengck23 [placeholder] my approach and results](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/285516)
- [Deep Watershed Transform for Instance Segmentation](https://arxiv.org/pdf/1611.08303.pdf)
- https://github.com/min2209/dwt
- Segmentation Models Pytorch https://github.com/qubvel/segmentation_models.pytorch
- https://en.wikipedia.org/wiki/Distance_transform
- https://stackoverflow.com/questions/61716670/distance-transform-the-function-does-not-work-properly
- https://stackoverflow.com/questions/61204462/error-in-function-distancetransform-python-using-opencv-3-4-9

- https://github.com/MouseLand/cellpose
- https://github.com/YukangWang/TextField


# Questions
1. What are 1x1 convolutions for?