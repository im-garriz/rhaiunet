import time
import os
import random
import imgaug
from datetime import datetime
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision import transforms

from adabelief_pytorch import AdaBelief

from models.segmentors.unet import UNet
from models.segmentors.rdau_net import RDAU_NET
from models.segmentors.unetpp import UNetpp
from models.segmentors.unet_sharp import UnetSharp, UNetSharp2, UNetSharp_sinNP
from models.segmentors.rca_iunet import RCA_IUNET
from models.segmentors.segnet import SegNet, FCN8s, FCN16VGG, FCN32VGG
from models.segmentors.deeplabv3_xception import DeepLab

from common.hyperparameters import HyperparameterReader
from common.dataset_handler import load_dataset
from common.image_transformations import load_img_transforms, UnNormalize
from common.data_augmentation import load_data_augmentation_pipes
from common.utils import torch_dice_loss, check_experiments_folder, check_runs_folder, torch_contour_loss
from common.segmentation_metrics import get_evaluation_metrics, get_evaluation_metrics_sections
from common.progress_logger import ProgressBar
from common.utils import generate_output_img


class UnetTrainer:

    def __init__(self, hyperparams_file):
        

        self.experiment_folder = check_experiments_folder()

        hyperparameter_loader = HyperparameterReader(hyperparams_file)
        self.parameter_dict = hyperparameter_loader.load_param_dict()

        self.LOG("Launching UnetTrainer...")
        self.LOG("Hyperparameters succesfully read from {hyperparams_file}:")
        for key, val in self.parameter_dict.items():
            self.LOG(f"\t{key}: {val}")
            
        self.set_random_seed(self.parameter_dict["random_seed"])

        if self.parameter_dict['device'] is False:
            self.parameter_dict["device"] = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        self.LOG("Found device: {}".format(self.parameter_dict["device"]))

        transforms_dict = load_img_transforms()
        self.LOG("Transforms dict load succesfully")

        augmentation_dict = load_data_augmentation_pipes(data_aug=self.parameter_dict["data_augmentation"], grayscale=self.parameter_dict["grayscale"])
        self.LOG("Data augmentation dict load succesfully")

        self.parameter_dict["transforms"] = transforms_dict
        self.parameter_dict["augmentation_pipelines"] = augmentation_dict

        self.dataset = load_dataset(self.parameter_dict)
        self.LOG("Dataset load succesfully")
        

        if self.parameter_dict["net"] == "UNet":
            self.model = UNet(1, 1, bilinear=self.parameter_dict["bilinear"]).to(self.parameter_dict["device"])
            self.model.init_weights()
        elif self.parameter_dict["net"] == "RDAUNet":
            self.model = RDAU_NET().to(self.parameter_dict["device"])
            self.model.init_weights()
        elif self.parameter_dict["net"] == "UNet2":
            self.model = UNet2(1, 1).to(self.parameter_dict["device"])
            self.model.init_weights()
        elif self.parameter_dict["net"] == "UNetpp":
            self.model = UNetpp(1).to(self.parameter_dict["device"])
            self.model.init_weights()
        elif self.parameter_dict["net"] == "UnetSharp":
            self.model = UnetSharp(nc=self.parameter_dict["nc"], pooling=self.parameter_dict["pooling"],
                                   dropout=self.parameter_dict["dropout"], block_size=self.parameter_dict["block_size"]).to(self.parameter_dict["device"])
            self.model.init_weights()
        elif self.parameter_dict["net"] == "RCAIUNet":
            self.model = RCA_IUNET(1, 16).to(self.parameter_dict["device"])
            self.model.init_weights()
        elif self.parameter_dict["net"] == "SegNet_VGG19":
            self.model = SegNet(pretrained=True).to(self.parameter_dict["device"])
            #self.model.init_weights()
        elif self.parameter_dict["net"] == "UNetSharp2":
            self.model = UNetSharp2(nc=self.parameter_dict["nc"], pooling=self.parameter_dict["pooling"],
                                    dropout=self.parameter_dict["dropout"], dropout_p=self.parameter_dict["dropout_p"],
                                    block_size=self.parameter_dict["block_size"]).to(self.parameter_dict["device"])
        elif self.parameter_dict["net"] == "UNetSharp_noSP":
            self.model = UNetSharp_sinNP(nc=self.parameter_dict["nc"], pooling=self.parameter_dict["pooling"],
                                    dropout=self.parameter_dict["dropout"], dropout_p=self.parameter_dict["dropout_p"],
                                    block_size=self.parameter_dict["block_size"]).to(self.parameter_dict["device"])
            self.model.init_weights()
        elif self.parameter_dict["net"] == "FCN8s":
            self.model = FCN8s(pretrained=True).to(self.parameter_dict["device"])
        elif self.parameter_dict["net"] == "FCN16VGG":
            self.model = FCN16VGG(pretrained=True).to(self.parameter_dict["device"])
        elif self.parameter_dict["net"] == "FCN32VGG":
            self.model = FCN32VGG(pretrained=True).to(self.parameter_dict["device"])
            
        elif self.parameter_dict["net"] == "DeepLabV3_Xception":
            self.model = DeepLab(backbone='xception', output_stride=16).to(self.parameter_dict["device"])
        elif self.parameter_dict["net"] == "DeepLabV3_ResNet50":
            self.model = torch.hub.load('pytorch/vision:v0.11.1', 'deeplabv3_resnet50',
                                        pretrained=False, num_classes=1).to(self.parameter_dict["device"])
        elif self.parameter_dict["net"] == "DeepLabV3_ResNet101":
            self.model = torch.hub.load('pytorch/vision:v0.11.1', 'deeplabv3_resnet101',
                                        pretrained=False, num_classes=1).to(self.parameter_dict["device"])
        elif self.parameter_dict["net"] == "DeepLabV3_MobileNet":
            self.model = torch.hub.load('pytorch/vision:v0.11.1', 'deeplabv3_mobilenet_v3_large',
                                        pretrained=False, num_classes=1).to(self.parameter_dict["device"])
        else:
            #TODO 
            pass
        if self.parameter_dict['multi_gpu']:
            #self.model = nn.DataParallel(self.model)
            self.model= nn.DataParallel(self.model, device_ids = [1, 2])

        self.LOG("Model {} load succesfully".format(self.parameter_dict["net"]))

        if self.parameter_dict["optimizer"] == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.parameter_dict["learning_rate"],
                                        betas=(0.9, 0.999))
        elif self.parameter_dict["optimizer"] == 'AdaBelief':
            self.optimizer = AdaBelief(self.model.parameters(), lr=self.parameter_dict["learning_rate"],
                                        betas=(0.9, 0.999))
        else:
            pass

        if self.parameter_dict["pretrained_weights"]:
            self.load_weights()
            self.LOG("Pretained weights load succesfully")  

        if self.parameter_dict["tensorboard"]:
            self.tb_runs_folder = check_runs_folder(self.experiment_folder.split('/')[-1])
            self.writer = SummaryWriter(f"{self.tb_runs_folder}")
        else:
            self.writer = None

        self.dsc_best =  -1


    @staticmethod
    def set_random_seed(random_seed):
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        imgaug.seed(random_seed)
        np.random.seed(random_seed)


    def load_weights(self, path_w=None):
        
        if path_w == None:
            try:
                self.model.load_state_dict(torch.load(self.parameter_dict["pretrained_weights_path"]))

            except:
                path = self.parameter_dict["pretrained_weights_path"]
                raise Exception(f"[E] Pretrained weights do not exist at {path} or they are not compatible")
        else:
            try:
                self.model.load_state_dict(torch.load(path_w))
            except:
                path = os.path.join('/home/imartinez/Code', path_w)
                raise Exception(f"[E] Pretrained weights do not exist at {path} or they are not compatible")


    def save_weights(self, best=False):

        path = os.path.join(self.experiment_folder, "weights")

        if not os.path.isdir(path):
            os.mkdir(path)

        if not best:
            torch.save(self.model.state_dict(), os.path.join(path, "last.pt"))  
        else:
            torch.save(self.model.state_dict(), os.path.join(path, "best.pt"))

    def update_learning_rate(self, epoch):
        
        PI = 3.14159
        n_epochs = self.parameter_dict["n_epochs"]
        initial_lr = self.parameter_dict["learning_rate"]
        lr = initial_lr * (np.cos(PI * epoch/n_epochs) + 1.) / 2
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr 
        

    @staticmethod
    def compute_loss(prediction, target, bce_weight=0., contour_loss_lambda=0.5):

        bce = F.binary_cross_entropy_with_logits(prediction, target)

        hard_sigmoid = nn.Hardsigmoid()
        prediction = hard_sigmoid(prediction)

        dice = torch_dice_loss(prediction, target)
        #adapted_dice = torch_dice_loss(prediction, target, adapt_values=True)
        
        if contour_loss_lambda > 0:
            contour = torch_contour_loss(prediction, target)
        else:
            contour = 0.
            
        loss = bce * bce_weight + contour * contour_loss_lambda +  dice * (1 - bce_weight - contour_loss_lambda) #+ 0.1 * torch.abs(dice - adapted_dice)
        
        return loss
    
    def update_contour_loss_lambda(self, epoch):
        
        lambda_ = self.parameter_dict['initial_lambda']
        
        if epoch <= self.parameter_dict['n_increment_epochs'] and epoch != 1:
             lambda_ += self.parameter_dict['lambda_increment_per_epoch']
        
        return lambda_ 


    def train_step(self, epoch):

        self.model.train()
        avg_loss = 0

        print("Train step")
        bar = ProgressBar(len(self.dataset.trainset_loader))
        
        for i, batched_sample in enumerate(self.dataset.trainset_loader):

            self.optimizer.zero_grad()

            images, masks, images_orig = batched_sample["image"].to(self.parameter_dict["device"]),\
                                         batched_sample["mask"].to(self.parameter_dict["device"]),\
                                         batched_sample["image_orig"].to(self.parameter_dict["device"])

            filenames = batched_sample["filename"]

            if self.parameter_dict['adversarial_training']:
                images.requires_grad = True

                
            if self.parameter_dict['net'] == 'UNetSharp2':
                output = self.model(images, images_orig)
            elif self.parameter_dict['net'] == 'DeepLabV3_ResNet101' or self.parameter_dict['net'] == 'DeepLabV3_ResNet50' or self.parameter_dict['net'] == 'DeepLabV3_MobileNet':
                output = self.model(images)['out']
            else:
                output = self.model(images)
                
            contour_loss_lambda = self.update_contour_loss_lambda(epoch)

            loss = self.compute_loss(output, masks, bce_weight=self.parameter_dict['bce_weight'], contour_loss_lambda=contour_loss_lambda)

            avg_loss += loss.item()

            if self.parameter_dict['adversarial_training']:
                loss.backward(retain_graph=True)
                images_grad = images.grad.data
                perturbed_images = self.fgsm_attack(images, images_grad, epsilon=self.parameter_dict['epsilon'])
                if self.parameter_dict['net'] == 'UNetSharp2':
                    output = self.model(perturbed_images, images_orig)
                elif self.parameter_dict['net'] == 'DeepLabV3_ResNet101' or self.parameter_dict['net'] == 'DeepLabV3_ResNet50' or self.parameter_dict['net'] == 'DeepLabV3_MobileNet':
                    output = self.model(images)['out']
                else:
                    output = self.model(perturbed_images)
                
                loss = self.compute_loss(output, masks)
            
            loss.backward()
            self.optimizer.step()

            bar.step_bar()

        return avg_loss / len(self.dataset.trainset_loader)

    def val_step(self):

        self.model.eval()

        print("Validation step")
        bar = ProgressBar(len(self.dataset.valset_loader))

        with torch.no_grad():
            avg_loss = 0
            
            for i, batched_sample in enumerate(self.dataset.valset_loader):

                images, masks, images_orig = batched_sample["image"].to(self.parameter_dict["device"]),\
                                             batched_sample["mask"].to(self.parameter_dict["device"]),\
                                             batched_sample["image_orig"].to(self.parameter_dict["device"])

                
                if self.parameter_dict['net'] == 'UNetSharp2':
                    output = self.model(images, images_orig)
                elif self.parameter_dict['net'] == 'DeepLabV3_ResNet101' or self.parameter_dict['net'] == 'DeepLabV3_ResNet50' or self.parameter_dict['net'] == 'DeepLabV3_MobileNet':
                    output = self.model(images)['out']
                else:
                    output = self.model(images)

                loss = self.compute_loss(output, masks)

                avg_loss += loss.item()

                bar.step_bar()

        return avg_loss / len(self.dataset.valset_loader)

    def train(self):

        self.LOG("Starting training the model...")

        for epoch in range(1, self.parameter_dict["n_epochs"]+1):

            self.LOG(f"Starting epoch {epoch}")
            self.update_learning_rate(epoch)

            start = time.time()
            train_loss = self.train_step(epoch)
            val_loss = self.val_step()
            end = time.time()
            elapsed = end - start

            msg = f"Epoch {epoch} finished -- Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f} -- Elapsed time: {elapsed:.1f}s"
            print(msg + "\n")
            self.LOG(msg)

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)

            metrics = get_evaluation_metrics(self.writer, epoch, self.dataset.valset_loader, self.model,
                                    self.parameter_dict["device"], writer=self.writer,
                                    SAVE_SEGS=False, N_EPOCHS_SAVE=10, folder=f"{self.experiment_folder}/segmentations",
                                    grayscale=self.parameter_dict["grayscale"], model=self.parameter_dict['net'])

            self.save_weights()
            self.LOG(f"Last weights saved at epoch {epoch}")

            if metrics.dice > self.dsc_best:

                self.LOG(f"New best value of DSC reach: {metrics.dice:.4f} (last: {self.dsc_best:.4f})")
                self.dsc_best = metrics.dice
                self.save_weights(best=True)
                self.LOG(f"Best weights saved at epoch {epoch}")

            self.LOG_METRICS(metrics, epoch, train_loss, val_loss)

    def test(self):

        #weights = self.parameter_dict['pretrained_weights_path']
        #weights.replace('')
        self.load_weights('/home/imartinez/Code/experiments_arquitectura/exp11/weights/last.pt')
        #self.load_weights('/home/imartinez/Code/_experiments_/exp7/weights/last.pt')
        #self.load_weights()
        
        self.model.eval()

        print("[I] Evalutating the model...")

        metrics = get_evaluation_metrics(None, -1, self.dataset.testset_loader, self.model,
                                    self.parameter_dict["device"], None, COLOR=True,
                                    SAVE_SEGS=True, N_EPOCHS_SAVE=5, folder=f"{self.experiment_folder}/segmentations", filename='avg_metrics_last.txt',
                                    grayscale=self.parameter_dict["grayscale"], model=self.parameter_dict['net'])

        print("\n----------------------------------------------------------------------------")
        print("EVALUTAION RESULTS ON TEST SET (LAST):")
        print("\tCCR: {:.4f}".format(metrics.CCR))
        print("\tPrecision: {:.4f}".format(metrics.precision))
        print("\tRecall: {:.4f}".format(metrics.recall))
        print("\tSensibility: {:.4f}".format(metrics.sensibility))
        print("\tSpecifity: {:.4f}".format(metrics.specifity))
        print("\tF1 score: {:.4f}".format(metrics.f1_score))
        print("\tJaccard coef: {:.4f}".format(metrics.jaccard))
        print("\tDSC coef: {:.4f}".format(metrics.dice))
        print("\tROC-AUC: {:.4f}".format(metrics.roc_auc))
        print("\tPrecision-recall AUC: {:.4f}".format(metrics.precision_recall_auc))
        print("\tHausdorf error: {:.4f}".format(metrics.hausdorf_error))
        print("----------------------------------------------------------------------------")
        print(f"Segmentations saved at {self.experiment_folder}/segmentations")
        
        self.test_per_area()
        
        return
        
        best_w = os.path.join(self.experiment_folder, "weights", 'best.pt')
        self.load_weights(best_w)
        
        metrics = get_evaluation_metrics(None, -1, self.dataset.testset_loader, self.model,
                                    self.parameter_dict["device"], None, COLOR=True,
                                    SAVE_SEGS=False, N_EPOCHS_SAVE=5, folder=f"{self.experiment_folder}/segmentations", filename='avg_metrics_best.txt',
                                    model=self.parameter_dict['net'])

        print("\n----------------------------------------------------------------------------")
        print("EVALUTAION RESULTS ON TEST SET (BEST):")
        print("\tCCR: {:.4f}".format(metrics.CCR))
        print("\tPrecision: {:.4f}".format(metrics.precision))
        print("\tRecall: {:.4f}".format(metrics.recall))
        print("\tSensibility: {:.4f}".format(metrics.sensibility))
        print("\tSpecifity: {:.4f}".format(metrics.specifity))
        print("\tF1 score: {:.4f}".format(metrics.f1_score))
        print("\tJaccard coef: {:.4f}".format(metrics.jaccard))
        print("\tDSC coef: {:.4f}".format(metrics.dice))
        print("\tROC-AUC: {:.4f}".format(metrics.roc_auc))
        print("\tPrecision-recall AUC: {:.4f}".format(metrics.precision_recall_auc))
        print("\tHausdorf error: {:.4f}".format(metrics.hausdorf_error))
        print("----------------------------------------------------------------------------")
        #print(f"Segmentations saved at {self.experiment_folder}/segmentations")
        
        
    def test_per_area(self, load=False):

        #weights.replace('')
        if load:
            self.load_weights()
            
        self.model.eval()

        print("\n[I] Evalutating the model per areas...")

        get_evaluation_metrics_sections(None, -1, self.dataset.testset_loader, self.model,
                                self.parameter_dict["device"], None, COLOR=True,
                                SAVE_SEGS=False, N_EPOCHS_SAVE=5, folder=f"{self.experiment_folder}/segmentations", filename='avg_metrics_per_area.txt',
                                grayscale=self.parameter_dict["grayscale"], model=self.parameter_dict['net'])

        print("Done\n")


        

    
    @staticmethod
    def fgsm_attack(image, data_grad, epsilon=0.01):

        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, -1, 1)
        # Return the perturbed image
        return perturbed_image


    def generate_adversarial_examples(self):

        self.load_weights()
        self.model.eval()

        trans = transforms.ToPILImage()

        for i, batched_sample in enumerate(self.dataset.testset_loader):

            images, masks = batched_sample["image"].to(self.parameter_dict["device"]),\
                            batched_sample["mask"].to(self.parameter_dict["device"])

            filenames = batched_sample["filename"]

            images.requires_grad = True

            output = self.model(images)

            loss = self.compute_loss(output, masks)

            self.model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            perturbed_data = self.fgsm_attack(images, images.grad.data)
            output = self.model(perturbed_data)

            hard_sigmoid = nn.Hardsigmoid()
            output = hard_sigmoid(output)

            for j in range(perturbed_data.shape[0]):
                image, mask = perturbed_data[j].to("cpu"), masks[j].to("cpu")
                segmentation = output[j].to("cpu")
                mask_save = trans(mask)
                name = filenames[j].split('/')[-1]

                image_save = trans(image.mul_(0.225).add_(0.485))
                segmentation_save = trans(segmentation)

                opencv_image = np.array(image_save)
                opencv_image = opencv_image[:, :, ::-1].copy()
                opencv_gt = np.array(mask_save)
                opencv_segmentation = np.array(segmentation_save)

                save_image = generate_output_img(opencv_image, opencv_gt, opencv_segmentation)
                cv2.imwrite(os.path.join('/workspace/shared_files/pruebas', f"{name}"), save_image)

    def generate_augmented_examples(self):

        self.model.eval()
        from models.segmentors.dropblock_ import DropBlock2D

        trans = transforms.ToPILImage()

        for i, batched_sample in enumerate(self.dataset.trainset_loader):

            images, masks = batched_sample["image"].to(self.parameter_dict["device"]),\
                            batched_sample["mask"].to(self.parameter_dict["device"])

            filenames = batched_sample["filename"]

            drop_block = DropBlock2D(block_size=17, drop_prob=0.075)
            images = drop_block(images)

            
            for j in range(images.shape[0]):
                image, mask = images[j].detach().to('cpu'), masks[j].detach().to('cpu')

                name = filenames[j].split('/')[-1]

                image_save = trans(image.mul_(0.5).add_(0.5))

                opencv_image = np.array(image_save)

                cv2.imwrite(os.path.join('/home/imartinez/Code/aug/', f"{name}"), opencv_image)

                
    def LOG(self, msg):

        file = os.path.join(self.experiment_folder, "log.txt")

        if not os.path.isfile(file):
            with open(file, 'w') as f:
                pass

        with open(file, 'a') as f:

            timestamp = str(datetime.now()).split('.')[0]
            f.write(f"{timestamp}: {msg}\n")

    def LOG_METRICS(self, metrics, epoch, train_loss, val_loss):

        file = os.path.join(self.experiment_folder, "metrics.csv")

        if not os.path.isfile(file):
            with open(file, 'w') as f:
                f.write("epoch,train_loss,val_loss,ccr,precision,recall,sensibility,specifity,f1_score,"+
                       "jaccard_coef,dsc_coef,roc_auc,pr_auc,hausdorf_error\n")

        with open(file, 'a') as f:

            f.write(f"{epoch},{train_loss},{val_loss},{metrics.CCR},{metrics.precision},"
                    f"{metrics.recall},{metrics.sensibility},{metrics.specifity},{metrics.f1_score},"
                    f"{metrics.jaccard},{metrics.dice},{metrics.roc_auc},{metrics.precision_recall_auc},"
                    f"{metrics.hausdorf_error}\n")


if __name__ == '__main__':

    unet_trainer = UnetTrainer("hyperparameters.yaml")