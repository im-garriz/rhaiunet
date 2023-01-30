import torch
import torch.nn as nn
import os
import numpy as np
import cv2


def weights_init(m):

    classname = m.__class__.__name__

    no_init = ["OutConv", "DoubleConv", "InceptionConvolutionLayer"]

    if classname.find('Conv') != -1:
        if classname not in no_init:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def torch_dice_loss(pred, target, smooth=1., adapt_values=False):

    pred = pred.contiguous()
    target = target.contiguous()

    if adapt_values:
        pred[pred >= 0.5] = 1.0
        pred[pred < 0.5] = 0.0

    loss = (1 - ((2. * (pred * target).sum(dim=2).sum(dim=2) + smooth) /
            (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def check_experiments_folder():

    if not os.path.isdir("experiments"):
        os.mkdir("experiments")
        os.mkdir("experiments/exp1")
        return "experiments/exp1"
    else:
        numbers = [int(x.replace("exp", ""))
                   for x in os.listdir("experiments")]
        if len(numbers) > 0:
            n_folder = max(numbers)+1
        else:
            n_folder = 1

        os.mkdir(f"experiments/exp{n_folder}")
        return f"experiments/exp{n_folder}"


def check_runs_folder(exp_folder):

    if not os.path.isdir("runs"):
        os.mkdir("runs")

    os.mkdir(f"runs/{exp_folder}")
    return f"runs/{exp_folder}/{exp_folder}"


def generate_output_img(image, gt, segmentation):

    GT_COLOUR = (0., 1., 0.)
    SEG_COLOUR = (0., 0., 1.)
    GT_SEG_COLOUR = (0., 1., 1.)
    ALPHA = 0.57
    ALPHA2 = 0.

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if np.max(gt) > 1.:
        gt = gt.astype(np.float32)
        gt /= 255.

    if np.max(segmentation) > 1.:
        segmentation = segmentation.astype(np.float32)
        segmentation /= 255.

    binary_segmentation = np.zeros_like(segmentation)
    binary_segmentation[segmentation >= 0.5] = 1

    gt_seg_intersect_mask = gt * binary_segmentation

    paint_mask = np.zeros_like(image)
    paint_mask[gt_seg_intersect_mask == 1., :] = GT_SEG_COLOUR
    paint_mask[(gt == 1.) & (gt_seg_intersect_mask == 0.)] = GT_COLOUR
    paint_mask[(binary_segmentation == 1.) & (
        gt_seg_intersect_mask == 0.)] = SEG_COLOUR
    paint_mask[(paint_mask[:, :, 2] == 0.) & (
        binary_segmentation == 1.), :] = SEG_COLOUR

    image_painted_with_segs = np.copy(image)
    cond = (binary_segmentation == 1.) | (gt == 1.)
    image_painted_with_segs[cond, :] = ALPHA * \
        image_painted_with_segs[cond, :] + (1-ALPHA) * paint_mask[cond] * 255.0

    ########################################################################################

    segmentation = (segmentation * 255).astype(np.uint8)
    heatmap_seg = cv2.applyColorMap(segmentation, cv2.COLORMAP_JET)

    heatmap_image = ALPHA2 * image + (1-ALPHA2) * heatmap_seg

    image_painted_with_segs = cv2.resize(image_painted_with_segs, (512, 512))
    heatmap_image = cv2.resize(heatmap_image, (512, 512))

    concated_images = np.hstack([image_painted_with_segs, heatmap_image])

    return concated_images


def get_model_outputs(net, model, images, images_orig=None):
    """
    Function that obtains the output of the model.
    As depending on the implementation of the model and the type of
    model the inputs vary and also the way to obtain the outputs,
    a separate function is used.
    """
    if net == 'RHAIUNet' or net == 'UNetSharp2':
        output = model(images, images_orig)
    elif net == 'DeepLabV3_ResNet101' or net == 'DeepLabV3_ResNet50' or net == 'DeepLabV3_MobileNet':
        output = model(images)['out']
    else:
        output = model(images)

    return output
