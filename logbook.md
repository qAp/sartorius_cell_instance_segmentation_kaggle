
## 2021-11-29
<img src="images/instance_direction_dataset_sample.png" width=900 height=700>
 
The Direction Net appears to be training.  The above plot shows the various 'images' involved in the training of this network.  The network input is the original image that's been gated by the semantic segmentation, so it's an image where the background is removed.  The area of each cell, or instance, is used as weights calculating the loss.  

uvec_0 and uvec_1 are the 0th and 1st components of the noramlised gradient of distance transform.  It can be seen, from how the colour changes from red to blue, that uvec_0 changes along the y-direction, while uvec_1 changes along the x-direction.  It's important to understand that this should always be observed, regardless of the how the image is rotated.  A mistake I have made is to pre-compute these components based on the original image and simply treat them like other masks, like semantic segmentation and instance area.  This is erroneous because whilst the semantic segmentation and instance area of a pixel are invariant under image rotation, the components of the normalised gradient are not.  Therefore, they should be computed from the distance transform *after* any augmentation transforms.  This was confirmed in my experiments, where, when there is no data augmentation, the validation loss decreases even when the gradients are pre-computed, because there is no image rotation.  But when there is image rotation, the validation loss only decrease when the gradients are computed after the image rotation.

<img src="images/direction_net_val_loss.png" width=900>

About the learning rate.  The authors used 1e-5 for 20 epochs.  I find that works; the validation loss decreases.  However, when I experimented with a one-cycle learning rate scheduler with a maximum learning rate of 1e-2, the validation loss does not really go down.  

cosmic-serenity-5 uses a constant learning rate of 1e-5 for 100 epochs.  

The following experiments include additional heavy data augmentations, and it's clear that the validation loss is lower for these.  rare-meadow-17 uses a maximum one-cycle learning rate of 1e-2, while happy-morning-18 uses 1e-3.  It's interesting that the validation loss decreases but suddenly start to increase, tending towards the same value as rare-meadow-17.  woven-frog-16 uses the same learning rate settings as the paper, a constant value of 1e-5 for 20 epochs.  

It appears that whether data augmentation is used or not, the validation loss tends to the same value.  It's just that in the case where data augmentation is used, it tends towards it from below, while without augmentation, it tends from above.  One apparent exception is woven-frog-16, and it would be interesting to see if it also tends to the same value when trained for more epochs.  

<img src="images/direction_net_train_loss.png" width=900>

The training loss doesn't appear to decrease with epochs trained.

## 2021-12-01
The distance transform is converted to watershed energy, which is just the distance transform sorted into a relatively small number of bins.  

Here are all the relevant maps that are considered in the overall workflow:
<img src="images/relevant_maps_upto_watershed_energy.png" width=1100>

The watershed energy ranges from 0 to 17.  Only larger cells have the highest energy levels.  The watershed energy levels are based on the discrete distance transform values.

## 2021-12-02
The watershed transform network (WTN) is implemented based on the description in the paper, but it's not a exact description, so will likely need to adjust the architecture later.  

The watershed loss has also been implemented based on the equation in the paper.  It's  not clear what the bar over variables $y$ and $t$ means, but given it's something like cross entropy, these are likely just $(1 - y)$ and $1 - t$, respectively.  Because of the pixel weights $w_p$, I first applied `nn.LogSoftmax` to the logits, then multiplied it with $w_p$.  Then, the energy weights $c_k$ are used to define a weighted negative log likelihood loss function with `nn.NLLoss`, which is then applied to the product of $w_p$ and the logits.  

The paper just says that errors in predicting the lower energies should be penalised more, without giving explicit values of $c_k$, so right now, these are just 17, 16, ..., 1. 

## 2021-12-03
Watershed transform network is training via `training/run_experiment.py`.

## 2021-12-04
How to cut the watershed energy to get individual instances is shown here https://github.com/min2209/dwt/blob/master/E2E/post_process.py.  Several functions from `skimage.morphology` are used.  

According to https://github.com/min2209/dwt/blob/master/matlab/generate_GT_cityscapes_unified.m, the authors used Matlab's [`bwdist`](https://www.mathworks.com/help/images/ref/bwdist.html) and [`imgradientxy`](https://www.mathworks.com/help/images/ref/imgradientxy.html) to compute the distance transform and gradient, respectively.  Might want to check if the Python functions I've been using correspond to these.

## 2021-12-05
To load a model from checkpoint and do inference, take the relevant `pl.LightningModule` and use the `load_from_checkpoint` method. Supply the same arguments as for the `__init__` method, plus the argument `checkpoint_path` that points to the checkpoint file produced by pytorch lightning during training, for example, by the `ModelCheckpoint` callback.  

Here's a pair of watershed energy ground truth and prediction by a WTN that's only been trained for 2 epochs:  
<img src="images/check_WTN_prediction.png" width=900>

When a Save & Run kaggle notebook times out, files saved in `/kaggle/working` are retained, and so if these are model checkpoints, they can be used to resume training. 

Kaggle's file system doesn't like filenames with `'='` in them.  When using pytorch lightning to save model, this could be avoided by setting `auto_insert_metric_name` to `False` *and* by using a format string for the filename as shown here, where 'epoch' directly preceeds '{' for example:
```
pl.callbacks.ModelCheckpoint(
        filename='epoch{epoch:03d}-val_loss{val_loss:.3f}',
        auto_insert_metric_name=False)
```

Regarding what to use in the inputs and targets for training the DN and WTN, it seems that there are two choices in general: to use model prediction-based targets, or to use competition target-based targets.  For example, the semantic segmentation can either come from the trained Unet, or from the competition target (the instance masks).  The semantic segmentation can in turn be used to obtain the normalised gradient distance transform and the watershed energy.  It seems that if a training target comes from the original competition target and if the input is based on model prediction, the learning task is most difficult, and therefore potentially most fruitful.  For example, using the Unet-generated semantic segmentation in the input of DN and using the competition target-generated normalised gradient distance transform as the target of DN might be a better learning task than if the input semantic segmentation were to be generated from the competition target.


## 2021-12-07
Sample prediction by Unet (Resnet152) after 299 epochs:

<img src="images/check_Unet_prediction.png" width=900>

## 2021-12-09

Current trained Unets do not appear to output good semantic segmentation results.  Data Science Bowl 2018 winning solution is studied, to see what's done there.  In particular, @selim's part: https://github.com/selimsef/dsb2018_topcoders/tree/master/selim.

Training targets.  They use 2-channel and 3-channel targets.  2-channel can for example be channel 0 being the nucleus mask and channel 1 being the touching borders mask. 3-channel is the same, but with channel 2 being everything else except channel 0 and 1.  Before being fed into the model, the targets are divided by 255, so they are of values between 0 and 1.

Inference.  If the image size is not a whole multiple of 32, it is padded to be.  It's also always padded by 16 on all sides.  Padding mode is symmetric.  For the pipeline with a 2-channel target and sigmoid output activation, the model output is of shape (height, width, 2).  This is averaged over a model ensemble and over 8 testtime augmentation predictions. 

Loss. For 2-channel target, the loss for channel 0 and channel 1 are computed using the same loss function, and then summed together.  The loss function is `dice_coef_loss_bce`, which is a weighted sum of cross binary entropy loss and dice coefficient loss.  

Learning rate is low, 1e-5, no more than 1e-4.  The scheduler simply decreases the learning rate gradually.

Trained for 70 epochs.

## 2021-12-10  
It's not clear exactly how the "cell nuclei" masks and the "touching border" masks were generated by @selim for the Data Science Bowl 2018.  They were already saved in the 'masks_all' directory in the solution provided.  

A guess after reviewing the masks of several samples is that the "touching border" is likely the overlap between cells that are dilations of the original cells.  This is channel 1 in the masks.  Channel 0 are the original cell masks with the overlap in channel 1 subtracted away.  So, effectively it's like taking away areas that are slightly larger than the original overlapped areas, and storing them in another channel as a separate training target.  The motivation behind this might be that it makes it easier to separate cells that either overlap or are very close to each other.

The amount of dilation will likely affect how finely the model can segment the cells.  Given some cells are very small in this competition, a dilation that's too large probably won't work too well, because the overlap area taken away will be comparable to the size of the cells.  

## 2021-12-11
Finished coding up the lightning datamodule for 2/3-channel semantic segmentation. 

Two samples have been identified, which have overlap area larger than cell area, as if the normal cell and overlap channels have been switched:  
<img src="images/repeatedly_annotated_sample_overlaps.png" width=900>

It might be that these images have been annotated twice.  Notice that many cells have two almost identical borders drawn over them. Different annotations on the same cell obviously have a lot of overlap:  
<img src="images/repeatedly_annotated_sample_borders.png" width=900>

These samples are simply dropped from the Dataset.  The k-folds are not re-generated, because there are only two such samples.

In the context of semantic segmentation, the IOU metric's threshold value is the probability value above which a pixel is classified as true.  It's not the overlap between a cell's predicted mask and its ground truth mask, above which the prediction is considered true positive.  

Need to correct k-fold generation!

## 2021-12-12
Sample ground truth and prediction after ~400 epochs of training for cell nuclei and touching borders:

<img src="images/after_400epochs_cell_and_border_training.png" width=900>

The metric iou@0.95 has reached about 0.9.  It still looks as if many cells are stuck together. 

Things to do:
- [ ] Subtract "touch borders" channel from "cell nuclei" channel to get the final output semantic segmentation?
- [ ] Based on this Unet, train the DN and WTN, finalising their architectures and loss functions.
- [ ] Complete the post-processing to get instance segments.



## Continued at Issues
