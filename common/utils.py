import torch
import torch.nn as nn
import os
import numpy as np
import cv2

def merge_images_with_masks(images, masks):

    """
    Genera las imagenes de 4 canales que se pasan al discriminador (3 de la imagen original + 1 con
    la mascara de segmentacion

    :param images: imagenes
    :param masks: mascaras de segmentacion
    :return: tensor con imagenes de 4 canales
    """

    batch_size = images.shape[0]
    img_dim = images.shape[2]
    merged = torch.rand(batch_size, 2, img_dim, img_dim)

    for i in range(batch_size):
        merged[i] = torch.cat((images[i], masks[i]))

    return merged

def weights_init(m):

    """
    Inicializacion de los pesos de la red

    :param m: red
    :return:
    """
    classname = m.__class__.__name__

    no_init = ["OutConv", "DoubleConv", "InceptionConvolutionLayer"]

    if classname.find('Conv') != -1:
        if classname not in no_init:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def torch_dice_loss(pred, target, smooth = 1., adapt_values=False):

    pred = pred.contiguous()
    target = target.contiguous()

    if adapt_values:
        pred[pred >= 0.5] = 1.0
        pred[pred < 0.5] = 0.0

    loss = (1 - ((2. * (pred * target).sum(dim=2).sum(dim=2) + smooth) / \
            (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def torch_contour_loss(pred, target):
    
    #https://github.com/rosanajurdi/Perimeter_loss/blob/master/losses.py
    
    pred = pred.contiguous()
    target = target.contiguous()
    
    b, _, w, h = pred.shape
    cl_pred = contour(pred).sum(axis=(2,3))
    target_skeleton = contour(target).sum(axis=(2,3))
    big_pen: Tensor = (cl_pred - target_skeleton) ** 2
    contour_loss = big_pen / (w * h)

    return contour_loss.mean(axis=0)
   
def contour(x):
    '''
    Differenciable aproximation of contour extraction
    
    '''   
    min_pool_x = torch.nn.functional.max_pool2d(x*-1, (3, 3), 1, 1)*-1
    max_min_pool_x = torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1)
    contour = torch.nn.functional.relu(max_min_pool_x - min_pool_x)
    return contour

def check_experiments_folder():
        
    if not os.path.isdir("experiments"):
        os.mkdir("experiments")
        os.mkdir("experiments/exp1")
        return "experiments/exp1"
    else:
        numbers = [int(x.replace("exp", "")) for x in os.listdir("experiments")]
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
    paint_mask[(binary_segmentation == 1.) & (gt_seg_intersect_mask == 0.)] = SEG_COLOUR
    paint_mask[(paint_mask[:, :, 2] == 0.) & (binary_segmentation == 1.), :] = SEG_COLOUR

    image_painted_with_segs = np.copy(image)
    cond = (binary_segmentation == 1.) | (gt == 1.)
    image_painted_with_segs[cond, :] = ALPHA * image_painted_with_segs[cond, :] + (1-ALPHA) * paint_mask[cond] * 255.0

    ########################################################################################

    segmentation = (segmentation * 255).astype(np.uint8)
    heatmap_seg = cv2.applyColorMap(segmentation, cv2.COLORMAP_JET)

    heatmap_image = ALPHA2 * image + (1-ALPHA2) * heatmap_seg

    image_painted_with_segs = cv2.resize(image_painted_with_segs, (512, 512))
    heatmap_image = cv2.resize(heatmap_image, (512, 512))

    concated_images = np.hstack([image_painted_with_segs, heatmap_image])
    
    return concated_images


if __name__ == '__main__':
    import cv2
    import numpy as np

    images_path = ["/home/inaki/shared_files/Dataset_TFM/images/BUSI/benign (18).png", "/home/inaki/shared_files/Dataset_TFM/images/DatasetB/benign_000019.png"]

    if not os.path.isdir("borrar"):
        os.mkdir('borrar')

    for i, img_f in enumerate(images_path):
        img = cv2.imread(img_f)

        img = cv2.resize(img, (128, 128))

        img_filtered = lee_filter(img,  2)

        save_img = np.hstack([img, img_filtered])

        cv2.imwrite(os.path.join('borrar', f'{i}.png'), save_img)