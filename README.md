
# A solution to the Sartorius Cell Instance Segmentation Kaggle
https://www.kaggle.com/c/sartorius-cell-instance-segmentation


# Solution summary
1. Semantic segmentation by Unet.
2. Instance segmentation by further processing of semantic segmentation with [Deep Watershed Transform](https://arxiv.org/pdf/1611.08303.pdf).

Deep Watershed Transform Network:  
<img src="images/direction_net.png" width=900 height=350>

# Semantic segmentation training
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