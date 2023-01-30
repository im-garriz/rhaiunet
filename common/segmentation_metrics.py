import seg_metrics.seg_metrics as sg
from tqdm import tqdm
import contextlib
from skimage import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from common.utils import generate_output_img
from hausdorff import hausdorff_distance
import numpy as np
import torch
from torchvision import transforms
import cv2
import os
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from common.utils import *
import warnings
warnings.filterwarnings("ignore")
# from common.progress_logger import ProgressLogger


class SegmentationEvaluationMetrics:

    def __init__(self, CCR, precision, recall, sensibility, specifity, f1_score,
                 jaccard, dice, roc_auc, precision_recall_auc, hausdorf_error, usr, osr):
        self.CCR = CCR
        self.precision = precision
        self.recall = recall
        self.sensibility = sensibility
        self.specifity = specifity
        self.f1_score = f1_score
        self.jaccard = jaccard
        self.dice = dice
        self.roc_auc = roc_auc
        self.precision_recall_auc = precision_recall_auc
        self.hausdorf_error = hausdorf_error
        self.osr = osr
        self.usr = usr


def compute_jaccard_dice_coeffs(mask1, mask2):
    """Calculates the dice coefficient for the images"""

    mask1 = np.asarray(mask1).astype(np.bool)
    mask2 = np.asarray(mask2).astype(np.bool)

    if mask1.shape != mask2.shape:
        raise ValueError(
            "Shape mismatch: mask1 and mask2 must have the same shape.")

    mask1 = mask1 > 0.5
    mask2 = mask2 > 0.5

    im_sum = mask1.sum() + mask2.sum()

    if im_sum == 0:
        return 1.0, 1.0

    intersection = np.logical_and(mask1, mask2).sum()
    union = im_sum - intersection

    return intersection / union, 2. * intersection / im_sum


def compute_hausdorff_dist(im1, im2):
    """Calculates the jaccard coefficient for the images"""

    im1 = np.asarray(im1).astype(np.int32)
    im2 = np.asarray(im2).astype(np.int32)

    im1 = im1.reshape(im1.shape[1], im1.shape[2])
    im2 = im2.reshape(im2.shape[1], im2.shape[2])

    if im1.shape != im2.shape:
        raise ValueError(
            "Shape mismatch: im1 and im2 must have the same shape.")

    distance = hausdorff_distance(im1, im2, distance="euclidean")
    return distance


def get_conf_mat(prediction, groundtruth):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""

    prediction = prediction > 0.5
    groundtruth = groundtruth > 0.5

    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))

    return FP, FN, TP, TN


def get_evaluation_metrics(logger, epoch, dataloader, segmentor, DEVICE, writer=None, SAVE_SEGS=False, COLOR=True,
                           N_EPOCHS_SAVE=10, folder="", filename='avg_metrics.txt', grayscale=True, model=''):

    if not os.path.isdir(folder):
        os.mkdir(folder)

    if not epoch == -1:
        save_folder = os.path.join(folder, f"epoch_{epoch}")
    else:
        save_folder = os.path.join(folder, "segmentations")

    if SAVE_SEGS and (epoch % N_EPOCHS_SAVE == 0 or epoch == -1):
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)

    ccrs = []

    precisions = []
    recalls = []

    sensibilities = []
    specifities = []

    f1_scores = []

    jaccard_coefs = []
    dice_coeffs = []

    roc_auc_coeffs = []
    precision_recall_auc_coeffs = []

    hausdorf_errors = []

    osrs = []
    usrs = []

    segmentor.eval()

    with open(os.path.join(folder, 'metrics.csv'), 'w') as file:
        file.write(
            'image_id,ccr,precision,recall,specififty,f1_score,jaccard,dsc,roc_auc,pr_auc,usr,osr,hausdorff_error\n')

    with torch.no_grad():

        # for i, batched_sample in enumerate(dataloader):
        for batched_sample in tqdm(dataloader, desc="Average metrics", unit=' batches', ncols=100):

            images, masks, filenames, images_orig = batched_sample["image"].to(DEVICE), batched_sample["mask"].to(DEVICE), \
                batched_sample["filename"], batched_sample["image_orig"].to(
                    DEVICE)

            hard_sigmoid = nn.Hardsigmoid()

            """
            if model == 'UNetSharp2':
                segmentations = hard_sigmoid(segmentor(images, images_orig))
            elif model == 'DeepLabV3_ResNet101' or model == 'DeepLabV3_ResNet50' or model == 'DeepLabV3_MobileNet':
                segmentations = hard_sigmoid(segmentor(images)['out'])
            else:
                segmentations = hard_sigmoid(segmentor(images))"""

            segmentations = hard_sigmoid(get_model_outputs(
                model, segmentor, images, images_orig))

            segmentation_values = segmentations
            segmentations = torch.autograd.Variable(
                (segmentations > 0.5).float())

            trans = transforms.ToPILImage()

            for j in range(images.shape[0]):
                image, mask = images[j].to("cpu"), masks[j].to("cpu")
                image_orig = images_orig[j].to("cpu")
                segmentation = segmentations[j].to("cpu")
                segmentation_val = segmentation_values[j].to("cpu")
                name = filenames[j].split('/')[-1]

                FP, FN, TP, TN = get_conf_mat(
                    segmentation.numpy(), mask.numpy())

                ccr = np.divide(TP + TN, FP + FN + TP + TN)

                precision = np.divide(TP, TP + FP)
                recall = np.divide(TP, TP + FN)

                sensibility = np.divide(TP, TP + FN)
                specifity = np.divide(TN, TN + FP)

                jaccard_coef, f1_score = compute_jaccard_dice_coeffs(
                    segmentation.numpy(), mask.numpy())
                dice_coeff = f1_score

                usr = np.divide(FN, TP + FN)
                osr = np.divide(FP, TP + FN)

                mask_labels = mask.numpy().ravel().astype(np.int32)
                segmentation_labels = segmentation.numpy().ravel()
                fpr, tpr, _ = roc_curve(mask_labels, segmentation_labels)
                roc_auc = auc(fpr, tpr)

                precision_values, recall_values, _ = precision_recall_curve(
                    mask_labels, segmentation_labels)
                precision_recall_auc = auc(recall_values, precision_values)

                hausdorf_error_ = compute_hausdorff_dist(
                    segmentation.numpy(), mask.numpy())
                hausdorf_error = metrics.hausdorff_distance(
                    segmentation.numpy(), mask.numpy())

                # print(mask.numpy())

                # cv2.imwrite(mask_file, 255 * mask.numpy().reshape((128, 128)))
                # cv2.imwrite(seg_file, 255 * segmentation.numpy().reshape((128, 128)))

                if np.max(mask.numpy()) != 0 or np.max(segmentation.numpy()) != 0:

                    # try:
                    _metrics = sg.write_metrics(labels=[1], gdth_img=mask.numpy().reshape((128, 128)),
                                                pred_img=segmentation.numpy().reshape((128, 128)),
                                                # csv_file='a.csv',
                                                metrics=['dice', 'hd95'],
                                                verbose=False)
                    # __dice = _metrics['dice']
                    __hd95 = _metrics[0]['hd95'][0]
                    hausdorf_error = __hd95
                    if hausdorf_error < 99999999:
                        hausdorf_errors.append(hausdorf_error)

                else:
                    # except:

                    #    print(i, j)
                    __hd95 = -1

                # print(_metrics)
                # print(__hd95)

                with open(os.path.join(folder, 'metrics.csv'), 'a') as file:
                    file.write(
                        f'{name},{ccr},{precision},{recall},{specifity},{f1_score},{jaccard_coef},{dice_coeff},{roc_auc},{precision_recall_auc},{usr},{osr},{hausdorf_error}\n')

                ccrs.append(ccr)
                precisions.append(precision)
                recalls.append(recall)
                sensibilities.append(sensibility)
                specifities.append(specifity)
                f1_scores.append(f1_score)
                jaccard_coefs.append(jaccard_coef)
                dice_coeffs.append(dice_coeff)
                roc_auc_coeffs.append(roc_auc)
                precision_recall_auc_coeffs.append(precision_recall_auc)

                usrs.append(usr)
                osrs.append(osr)

                if SAVE_SEGS and (epoch % N_EPOCHS_SAVE == 0 or epoch == -1):
                    #image_save = trans(image.mul_(0.5).add_(0.5))
                    image_save = trans(image_orig.mul_(0.5).add_(0.5))
                    mask_save = trans(mask)
                    segmentation_save = trans(segmentation)
                    segmentation_save_vals = trans(segmentation_val)

                    opencv_image = np.array(image_save).copy()

                    if not grayscale:
                        opencv_image = np.array(image_save)[:, :, 0].copy()
                    else:
                        opencv_image = np.array(image_save).copy()

                    # opencv_image = opencv_image[:, :, ::-1].copy()
                    opencv_gt = np.array(mask_save)
                    opencv_segmentation = np.array(segmentation_save)
                    opencv_segmentation_vals = np.array(segmentation_save_vals)

                    if not COLOR:
                        img = np.vstack(
                            (cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY), opencv_gt, opencv_segmentation))
                        cv2.imwrite(os.path.join(
                            save_folder, f"{name}.png"), img)
                    else:

                        save_image = generate_output_img(
                            opencv_image, opencv_gt, opencv_segmentation_vals)
                        # save_image = opencv_image

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(
                            save_image, f'DSC: {dice_coeff:.3f}', (512+50, 480), font, 1, (0, 0, 255), 2)
                        # save_image = cv2.resize(save_image, (512, 512))

                        cv2.imwrite(os.path.join(
                            save_folder, f"{name}"), save_image)

        ccrs = np.array(ccrs)[~np.isnan(np.array(ccrs))]
        precisions = np.array(precisions)[~np.isnan(np.array(precisions))]
        recalls = np.array(recalls)[~np.isnan(np.array(recalls))]
        sensibilities = np.array(sensibilities)[
            ~np.isnan(np.array(sensibilities))]
        specifities = np.array(specifities)[~np.isnan(np.array(specifities))]
        f1_scores = np.array(f1_scores)[~np.isnan(np.array(f1_scores))]
        jaccard_coefs = np.array(jaccard_coefs)[
            ~np.isnan(np.array(jaccard_coefs))]
        dice_coeffs = np.array(dice_coeffs)[~np.isnan(np.array(dice_coeffs))]
        roc_auc_coeffs = np.array(roc_auc_coeffs)[
            ~np.isnan(np.array(roc_auc_coeffs))]
        precision_recall_auc_coeffs = np.array(precision_recall_auc_coeffs)[
            ~np.isnan(np.array(precision_recall_auc_coeffs))]
        # [~np.isnan(np.array(hausdorf_errors))]
        hausdorf_errors = np.array(hausdorf_errors)
        usrs = np.array(usrs)[~np.isnan(np.array(usrs))]
        osrs = np.array(osrs)[~np.isnan(np.array(osrs))]

        mean_ccr = np.nansum(ccrs) / len(ccrs)
        std_ccr = np.std(np.array(ccrs))

        mean_precision = np.nansum(precisions) / len(precisions)
        std_precision = np.std(np.array(precisions))

        mean_recall = np.nansum(recalls) / len(recalls)
        std_recall = np.std(np.array(recalls))

        mean_sensibility = np.nansum(sensibilities) / len(sensibilities)
        std_sensibility = np.std(np.array(sensibilities))

        mean_specifity = np.nansum(specifities) / len(specifities)
        std_specifity = np.std(np.array(specifities))

        mean_f1_score = np.nansum(f1_scores) / len(f1_scores)
        std_f1_score = np.std(np.array(f1_scores))

        mean_jaccard_coef = np.nansum(jaccard_coefs) / len(jaccard_coefs)
        std_jaccard_coef = np.std(np.array(jaccard_coefs))

        mean_dice_coeff = np.nansum(dice_coeffs) / len(dice_coeffs)
        std_dice_coeff = np.std(np.array(dice_coeffs))

        mean_roc_auc = np.nansum(roc_auc_coeffs) / len(roc_auc_coeffs)
        std_roc_auc = np.std(np.array(roc_auc_coeffs))

        precision_recall_auc = np.nansum(
            precision_recall_auc_coeffs) / len(precision_recall_auc_coeffs)
        std_pr_auc = np.std(np.array(precision_recall_auc_coeffs))

        mean_hausdorf_error = np.nansum(hausdorf_errors) / len(hausdorf_errors)
        std_hausdorf_error = np.std(np.array(hausdorf_errors))

        mean_usr = np.nansum(usrs) / len(usrs)
        std_usr = np.std(np.array(usrs))

        mean_osr = np.nansum(osrs) / len(osrs)
        std_osr = np.std(np.array(osrs))

        segmentor.train()

        with open(os.path.join(folder, filename), 'w') as file:
            file.write(f"CCR: {mean_ccr:.4f} +- {std_ccr:.3f}\n")
            file.write(
                f"Precision: {mean_precision:.4f} +- {std_precision:.3f}\n")
            file.write(f"Recall: {mean_recall:.4f} +- {std_recall:.3f}\n")
            file.write(
                f"Specifity: {mean_specifity:.4f} +- {std_specifity:.3f}\n")
            file.write(
                f"F1 score: {mean_f1_score:.4f} +- {std_f1_score:.3f}\n")
            file.write(
                f"Jaccard: {mean_jaccard_coef:.4f} +- {std_jaccard_coef:.3f}\n")
            file.write(f"DSC: {mean_dice_coeff:.4f} +- {std_dice_coeff:.3f}\n")
            file.write(f"ROC AUC: {mean_roc_auc:.4f} +- {std_roc_auc:.3f}\n")
            file.write(
                f"PR AUC: {precision_recall_auc:.4f} +- {std_pr_auc:.3f}\n")
            file.write(
                f"Hausdorf error: {mean_hausdorf_error:.4f} +- {std_hausdorf_error:.3f}\n")
            file.write(f"USR: {mean_usr:.4f} +- {std_usr:.3f}\n")
            file.write(f"OSR: {mean_osr:.4f} +- {std_osr:.3f}\n")

        if writer is not None:
            writer.add_scalar("Metrics/ccr", mean_ccr, epoch)
            writer.add_scalar("Metrics/precision", mean_precision, epoch)
            writer.add_scalar("Metrics/recall", mean_recall, epoch)
            writer.add_scalar("Metrics/sensibility", mean_sensibility, epoch)
            writer.add_scalar("Metrics/specifity", mean_specifity, epoch)
            writer.add_scalar("Metrics/f1 score", mean_f1_score, epoch)
            writer.add_scalar("Metrics/jaccard idx", mean_jaccard_coef, epoch)
            writer.add_scalar("Metrics/dice coeff", mean_dice_coeff, epoch)
            writer.add_scalar("Metrics/roc-auc", mean_roc_auc, epoch)
            writer.add_scalar("Metrics/precision recall auc",
                              precision_recall_auc, epoch)
            writer.add_scalar("Metrics/hausdorf error",
                              mean_hausdorf_error, epoch)

        return SegmentationEvaluationMetrics(mean_ccr, mean_precision, mean_recall,
                                             mean_sensibility, mean_specifity, mean_f1_score, mean_jaccard_coef,
                                             mean_dice_coeff, mean_roc_auc, precision_recall_auc, mean_hausdorf_error, usr, osr)


###############################################################################################################################################

def get_evaluation_metrics_sections(logger, epoch, dataloader, segmentor, DEVICE, writer=None, SAVE_SEGS=False, COLOR=True,
                                    N_EPOCHS_SAVE=10, folder="", filename='avg_metrics_per_area.txt', grayscale=True, model=''):

    if not os.path.isdir(folder):
        os.mkdir(folder)

    if not epoch == -1:
        save_folder = os.path.join(folder, f"epoch_{epoch}")
    else:
        save_folder = os.path.join(folder, "segmentations")

    if SAVE_SEGS and (epoch % N_EPOCHS_SAVE == 0 or epoch == -1):
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)

    f1_scores_s1 = []
    f1_scores_s2 = []
    f1_scores_s3 = []
    f1_scores_s4 = []
    f1_scores_s5 = []

    dice_coeffs_s1 = []
    dice_coeffs_s2 = []
    dice_coeffs_s3 = []
    dice_coeffs_s4 = []
    dice_coeffs_s5 = []

    usr_s1 = []
    usr_s2 = []
    usr_s3 = []
    usr_s4 = []
    usr_s5 = []

    osr_s1 = []
    osr_s2 = []
    osr_s3 = []
    osr_s4 = []
    osr_s5 = []

    hd_s1 = []
    hd_s2 = []
    hd_s3 = []
    hd_s4 = []
    hd_s5 = []

    segmentor.eval()

    filename_per_point = 'metrics_xy.csv'

    with open(os.path.join(folder, filename_per_point), 'w') as file:
        file.write('DSC,JC,USR,OSR,hausdorff_erorr,Area\n')

    with torch.no_grad():

        # for i, batched_sample in enumerate(dataloader):
        for batched_sample in tqdm(dataloader, desc="Per-area interval metrics", unit=' batches', ncols=100):

            images, masks, filenames, images_orig = batched_sample["image"].to(DEVICE), batched_sample["mask"].to(DEVICE), \
                batched_sample["filename"], batched_sample["image_orig"].to(
                    DEVICE)

            hard_sigmoid = nn.Hardsigmoid()

            """
            if model == 'UNetSharp2':
                segmentations = hard_sigmoid(segmentor(images, images_orig))
            elif model == 'DeepLabV3_ResNet101' or model == 'DeepLabV3_ResNet50' or model == 'DeepLabV3_MobileNet':
                segmentations = hard_sigmoid(segmentor(images)['out'])
            else:
                segmentations = hard_sigmoid(segmentor(images))"""

            segmentations = hard_sigmoid(get_model_outputs(
                model, segmentor, images, images_orig))
            segmentation_values = segmentations
            segmentations = torch.autograd.Variable(
                (segmentations > 0.5).float())

            trans = transforms.ToPILImage()

            for j in range(images.shape[0]):
                image, mask = images[j].to("cpu"), masks[j].to("cpu")
                segmentation = segmentations[j].to("cpu")
                segmentation_val = segmentation_values[j].to("cpu")
                name = filenames[j].split('/')[-1]

                FP, FN, TP, TN = get_conf_mat(
                    segmentation.numpy(), mask.numpy())

                ccr = np.divide(TP + TN, FP + FN + TP + TN)

                precision = np.divide(TP, TP + FP)
                recall = np.divide(TP, TP + FN)

                sensibility = np.divide(TP, TP + FN)
                specifity = np.divide(TN, TN + FP)

                jaccard_coef, f1_score = compute_jaccard_dice_coeffs(
                    segmentation.numpy(), mask.numpy())
                dice_coeff = f1_score

                usr = np.divide(FN, TP + FN)
                osr = np.divide(FP, TP + FN)

                mask_labels = mask.numpy().ravel().astype(np.int32)
                segmentation_labels = segmentation.numpy().ravel()
                fpr, tpr, _ = roc_curve(mask_labels, segmentation_labels)
                roc_auc = auc(fpr, tpr)

                precision_values, recall_values, _ = precision_recall_curve(
                    mask_labels, segmentation_labels)
                precision_recall_auc = auc(recall_values, precision_values)

                hausdorf_error = metrics.hausdorff_distance(
                    segmentation.numpy(), mask.numpy())

                mask_save = trans(mask)
                opencv_gt = np.array(mask_save)
                area = np.sum(opencv_gt // 255)

                if np.max(mask.numpy()) != 0 or np.max(segmentation.numpy()) != 0:

                    # try:
                    _metrics = sg.write_metrics(labels=[1], gdth_img=mask.numpy().reshape((128, 128)),
                                                pred_img=segmentation.numpy().reshape((128, 128)),
                                                # csv_file='a.csv',
                                                metrics=['dice', 'hd95'],
                                                verbose=False)
                    # __dice = _metrics['dice']
                    __hd95 = _metrics[0]['hd95'][0]
                    hausdorf_error = __hd95

                    if hausdorf_error < 99999999:
                        if area == 0:
                            pass
                        elif area <= 150:

                            hd_s1.append(hausdorf_error)
                        elif area <= 500:

                            hd_s2.append(hausdorf_error)
                        elif area <= 2000:

                            hd_s3.append(hausdorf_error)
                        elif area <= 4000:

                            hd_s4.append(hausdorf_error)
                        else:

                            hd_s5.append(hausdorf_error)

                else:
                    # except:

                    #    print(i, j)
                    __hd95 = -1

                # print(_metrics)
                # print(__hd95)

                # print(f'F1 score: {f1_score} | DSC: {dice_coeff}')
                # print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}\n')
                with open(os.path.join(folder, filename_per_point), 'a') as file:
                    file.write(
                        f'{f1_score},{jaccard_coef},{usr},{osr},{hausdorf_error},{area}\n')

                if area == 0:
                    pass
                elif area <= 150:
                    f1_scores_s1.append(f1_score)
                    dice_coeffs_s1.append(jaccard_coef)
                    usr_s1.append(usr)
                    osr_s1.append(osr)
                    # hd_s1.append(hausdorf_error)
                elif area <= 500:
                    f1_scores_s2.append(f1_score)
                    dice_coeffs_s2.append(jaccard_coef)
                    usr_s2.append(usr)
                    osr_s2.append(osr)
                    # hd_s2.append(hausdorf_error)
                elif area <= 2000:
                    f1_scores_s3.append(f1_score)
                    dice_coeffs_s3.append(jaccard_coef)
                    usr_s3.append(usr)
                    osr_s3.append(osr)
                    # hd_s3.append(hausdorf_error)
                elif area <= 4000:
                    f1_scores_s4.append(f1_score)
                    dice_coeffs_s4.append(jaccard_coef)
                    usr_s4.append(usr)
                    osr_s4.append(osr)
                    # hd_s4.append(hausdorf_error)
                else:
                    f1_scores_s5.append(f1_score)
                    dice_coeffs_s5.append(jaccard_coef)
                    usr_s5.append(usr)
                    osr_s5.append(osr)
                    # hd_s5.append(hausdorf_error)

        f1_scores_s1 = np.array(f1_scores_s1)[
            ~np.isnan(np.array(f1_scores_s1))]
        f1_scores_s2 = np.array(f1_scores_s2)[
            ~np.isnan(np.array(f1_scores_s2))]
        f1_scores_s3 = np.array(f1_scores_s3)[
            ~np.isnan(np.array(f1_scores_s3))]
        f1_scores_s4 = np.array(f1_scores_s4)[
            ~np.isnan(np.array(f1_scores_s4))]
        f1_scores_s5 = np.array(f1_scores_s5)[
            ~np.isnan(np.array(f1_scores_s5))]

        dice_coeffs_s1 = np.array(dice_coeffs_s1)[
            ~np.isnan(np.array(dice_coeffs_s1))]
        dice_coeffs_s2 = np.array(dice_coeffs_s2)[
            ~np.isnan(np.array(dice_coeffs_s2))]
        dice_coeffs_s3 = np.array(dice_coeffs_s3)[
            ~np.isnan(np.array(dice_coeffs_s3))]
        dice_coeffs_s4 = np.array(dice_coeffs_s4)[
            ~np.isnan(np.array(dice_coeffs_s4))]
        dice_coeffs_s5 = np.array(dice_coeffs_s5)[
            ~np.isnan(np.array(dice_coeffs_s5))]

        hd_s1 = np.array(hd_s1)[~np.isnan(np.array(hd_s1))]
        hd_s2 = np.array(hd_s2)[~np.isnan(np.array(hd_s2))]
        hd_s3 = np.array(hd_s3)[~np.isnan(np.array(hd_s3))]
        hd_s4 = np.array(hd_s4)[~np.isnan(np.array(hd_s4))]
        hd_s5 = np.array(hd_s5)[~np.isnan(np.array(hd_s5))]

        def get_mean_and_std(data):

            mean = np.nansum(data) / len(data)
            std = np.std(np.array(data))

            return mean, std

        mean_f1_s1, std_f1_s1 = get_mean_and_std(f1_scores_s1)
        mean_f1_s2, std_f1_s2 = get_mean_and_std(f1_scores_s2)
        mean_f1_s3, std_f1_s3 = get_mean_and_std(f1_scores_s3)
        mean_f1_s4, std_f1_s4 = get_mean_and_std(f1_scores_s4)
        mean_f1_s5, std_f1_s5 = get_mean_and_std(f1_scores_s5)

        mean_dsc_s1, std_dsc_s1 = get_mean_and_std(dice_coeffs_s1)
        mean_dsc_s2, std_dsc_s2 = get_mean_and_std(dice_coeffs_s2)
        mean_dsc_s3, std_dsc_s3 = get_mean_and_std(dice_coeffs_s3)
        mean_dsc_s4, std_dsc_s4 = get_mean_and_std(dice_coeffs_s4)
        mean_dsc_s5, std_dsc_s5 = get_mean_and_std(dice_coeffs_s5)

        mean_usr_s1, std_usr_s1 = get_mean_and_std(usr_s1)
        mean_usr_s2, std_usr_s2 = get_mean_and_std(usr_s2)
        mean_usr_s3, std_usr_s3 = get_mean_and_std(usr_s3)
        mean_usr_s4, std_usr_s4 = get_mean_and_std(usr_s4)
        mean_usr_s5, std_usr_s5 = get_mean_and_std(usr_s5)

        mean_osr_s1, std_osr_s1 = get_mean_and_std(osr_s1)
        mean_osr_s2, std_osr_s2 = get_mean_and_std(osr_s2)
        mean_osr_s3, std_osr_s3 = get_mean_and_std(osr_s3)
        mean_osr_s4, std_osr_s4 = get_mean_and_std(osr_s4)
        mean_osr_s5, std_osr_s5 = get_mean_and_std(osr_s5)

        mean_hd_s1, std_hd_s1 = get_mean_and_std(hd_s1)
        mean_hd_s2, std_hd_s2 = get_mean_and_std(hd_s2)
        mean_hd_s3, std_hd_s3 = get_mean_and_std(hd_s3)
        mean_hd_s4, std_hd_s4 = get_mean_and_std(hd_s4)
        mean_hd_s5, std_hd_s5 = get_mean_and_std(hd_s5)

        segmentor.train()

    with open(os.path.join(folder, filename), 'w') as file:

        file.write(
            f"F1 score s1: {mean_f1_s1:.4f} +- {std_f1_s1:.3f} ({len(f1_scores_s1)})\n")
        file.write(
            f"F1 score s2: {mean_f1_s2:.4f} +- {std_f1_s2:.3f} ({len(f1_scores_s2)})\n")
        file.write(
            f"F1 score s3: {mean_f1_s3:.4f} +- {std_f1_s3:.3f} ({len(f1_scores_s3)})\n")
        file.write(
            f"F1 score s4: {mean_f1_s4:.4f} +- {std_f1_s4:.3f} ({len(f1_scores_s4)})\n")
        file.write(
            f"F1 score s5: {mean_f1_s5:.4f} +- {std_f1_s5:.3f} ({len(f1_scores_s5)})\n\n")

        file.write(
            f"JC s1: {mean_dsc_s1:.4f} +- {std_dsc_s1:.3f} ({len(dice_coeffs_s1)})\n")
        file.write(
            f"JC s2: {mean_dsc_s2:.4f} +- {std_dsc_s2:.3f} ({len(dice_coeffs_s2)})\n")
        file.write(
            f"JC s3: {mean_dsc_s3:.4f} +- {std_dsc_s3:.3f} ({len(dice_coeffs_s3)})\n")
        file.write(
            f"JC s4: {mean_dsc_s4:.4f} +- {std_dsc_s4:.3f} ({len(dice_coeffs_s4)})\n")
        file.write(
            f"JC s5: {mean_dsc_s5:.4f} +- {std_dsc_s5:.3f} ({len(dice_coeffs_s5)})\n\n")

        file.write(
            f"USR s1: {mean_usr_s1:.4f} +- {std_usr_s1:.3f} ({len(usr_s1)})\n")
        file.write(
            f"USR s2: {mean_usr_s2:.4f} +- {std_usr_s2:.3f} ({len(usr_s2)})\n")
        file.write(
            f"USR s3: {mean_usr_s3:.4f} +- {std_usr_s3:.3f} ({len(usr_s3)})\n")
        file.write(
            f"USR s4: {mean_usr_s4:.4f} +- {std_usr_s4:.3f} ({len(usr_s4)})\n")
        file.write(
            f"USR s5: {mean_usr_s5:.4f} +- {std_usr_s5:.3f} ({len(usr_s5)})\n\n")

        file.write(
            f"OSR s1: {mean_osr_s1:.4f} +- {std_osr_s1:.3f} ({len(osr_s1)})\n")
        file.write(
            f"OSR s2: {mean_osr_s2:.4f} +- {std_osr_s2:.3f} ({len(osr_s2)})\n")
        file.write(
            f"OSR s3: {mean_osr_s3:.4f} +- {std_osr_s3:.3f} ({len(osr_s3)})\n")
        file.write(
            f"OSR s4: {mean_osr_s4:.4f} +- {std_osr_s4:.3f} ({len(osr_s4)})\n")
        file.write(
            f"OSR s5: {mean_osr_s5:.4f} +- {std_osr_s5:.3f} ({len(osr_s5)})\n\n")

        file.write(
            f"HD s1: {mean_hd_s1:.4f} +- {std_hd_s1:.3f} ({len(hd_s1)})\n")
        file.write(
            f"HD s2: {mean_hd_s2:.4f} +- {std_hd_s2:.3f} ({len(hd_s2)})\n")
        file.write(
            f"HD s3: {mean_hd_s3:.4f} +- {std_hd_s3:.3f} ({len(hd_s3)})\n")
        file.write(
            f"HD s4: {mean_hd_s4:.4f} +- {std_hd_s4:.3f} ({len(hd_s4)})\n")
        file.write(
            f"HD s5: {mean_hd_s5:.4f} +- {std_hd_s5:.3f} ({len(hd_s5)})\n")
