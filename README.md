
# A solution to the Sartorius Cell Instance Segmentation Kaggle
https://www.kaggle.com/c/sartorius-cell-instance-segmentation

# Challenge to-do list
- [ ] Use different image tranformations for validation.
- [ ] Monitor validation loss, instead of validation metric, for callbacks.
- [ ] Keep the channel dimension of mask through training like in segmentation_models_pytorch?
- [x] Get segmentation training going.
- [ ] Generate normalised gradient of distance transform for training Direction Net.
- [ ] Construct Direction Net.

# Direction Net
<img src="images/direction_net.png" width=500 height=500>

# Notes on Deep Watershed Transform
## Instance level segmentation approaches
### Proposal based
### Deep structure models
### Template matching
### Recurrent networks
### CNN
### Proposal + recursion


# References
- [[ods.ai] topcoders, 1st place solution](https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741)
- [@hengck23 [placeholder] my approach and results](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/285516)
- [Deep Watershed Transform for Instance Segmentation](https://arxiv.org/pdf/1611.08303.pdf)
- https://github.com/min2209/dwt
- https://en.wikipedia.org/wiki/Distance_transform
- https://stackoverflow.com/questions/61716670/distance-transform-the-function-does-not-work-properly
- https://stackoverflow.com/questions/61204462/error-in-function-distancetransform-python-using-opencv-3-4-9

- https://github.com/MouseLand/cellpose
- https://github.com/YukangWang/TextField


# Questions
1. What are 1x1 convolutions for?