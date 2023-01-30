from common.utils import get_model_outputs
from common.segmentation_metrics import get_evaluation_metrics, get_evaluation_metrics_sections
from common.utils import torch_dice_loss, check_experiments_folder, check_runs_folder
from common.data_augmentation import load_data_augmentation_pipes
from common.dataset_handler import load_dataset
from common.hyperparameters import HyperparameterReader
from models.deeplabv3_xception import DeepLab
from models.segnet import SegNet, FCN8s, FCN16VGG, FCN32VGG
from models.rhaiunet import RHAIUNET
from models.ablation_exps.unet_sharp import UnetSharp, UNetSharp2, UNetSharp_sinNP
from models.unetpp import UNetpp
from models.rdau_net import RDAU_NET
from models.unet import UNet
from adabelief_pytorch import AdaBelief
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import time
import os
import random
import imgaug
from datetime import datetime
import numpy as np
from tqdm import tqdm

import torch
TORCH_VERSION = torch.__version__


class Trainer:

    def __init__(self, hyperparams_file):

        # Folder where all the results of the experiment will be stored.
        self.experiment_folder = check_experiments_folder()

        hyperparameter_loader = HyperparameterReader(hyperparams_file)
        self.parameter_dict = hyperparameter_loader.load_param_dict()

        self.LOG("Launching UnetTrainer...")
        self.LOG(f"Hyperparameters succesfully read from {hyperparams_file}:")
        for key, val in self.parameter_dict.items():
            self.LOG(f"\t{key}: {val}")

        self.set_random_seed(self.parameter_dict["random_seed"])

        # Device
        if self.parameter_dict['device'] is False:
            self.parameter_dict["device"] = torch.device(
                "cuda:0" if (torch.cuda.is_available()) else "cpu")
        elif self.parameter_dict['device'].startswith("cuda"):
            if not torch.cuda.is_available():
                dev = self.parameter_dict["device"]
                print(
                    f"{dev} selected as device but torch.cuda.is_available() returns False. Setting device='cpu'")
                self.parameter_dict["device"] = torch.device("cpu")

        self.LOG("Found device: {}".format(self.parameter_dict["device"]))

        #self.parameter_dict["transforms"] = load_img_transforms()
        #self.LOG("Transforms dict load succesfully")

        self.parameter_dict["augmentation_pipelines"] = load_data_augmentation_pipes(
            data_aug=self.parameter_dict["data_augmentation"], grayscale=self.parameter_dict["grayscale"])
        self.LOG("Data augmentation dict load succesfully")

        self.dataset = load_dataset(self.parameter_dict)
        self.LOG("Dataset load succesfully")

        # Load model
        if self.parameter_dict["net"] == "UNet":
            self.model = UNet(1, 1).to(
                self.parameter_dict["device"])
            self.model.init_weights()
        elif self.parameter_dict["net"] == "RDAUNet":
            self.model = RDAU_NET().to(self.parameter_dict["device"])
            self.model.init_weights()
        elif self.parameter_dict["net"] == "UNetpp":
            self.model = UNetpp(1).to(self.parameter_dict["device"])
            self.model.init_weights()
        elif self.parameter_dict["net"] == "UnetSharp":
            self.model = UnetSharp(nc=self.parameter_dict["nc"], pooling=self.parameter_dict["pooling"],
                                   dropout=self.parameter_dict["dropout"], block_size=self.parameter_dict["block_size"]).to(self.parameter_dict["device"])
            self.model.init_weights()
        elif self.parameter_dict["net"] == "RHAIUNet":
            self.model = RHAIUNET(nc=self.parameter_dict["nc"], pooling="Hartley").to(
                self.parameter_dict["device"])
            self.model.init_weights()

        elif self.parameter_dict["net"] == "SegNet_VGG19":
            self.model = SegNet(pretrained=True).to(
                self.parameter_dict["device"])
        elif self.parameter_dict["net"] == "FCN8s":
            self.model = FCN8s(pretrained=True).to(
                self.parameter_dict["device"])
        elif self.parameter_dict["net"] == "FCN16VGG":
            self.model = FCN16VGG(pretrained=True).to(
                self.parameter_dict["device"])
        elif self.parameter_dict["net"] == "FCN32VGG":
            self.model = FCN32VGG(pretrained=True).to(
                self.parameter_dict["device"])

        elif self.parameter_dict["net"] == "DeepLabV3_Xception":
            self.model = DeepLab(backbone='xception', output_stride=16).to(
                self.parameter_dict["device"])
        elif self.parameter_dict["net"] == "DeepLabV3_ResNet50":
            self.model = torch.hub.load('pytorch/vision:v0.11.1', 'deeplabv3_resnet50',
                                        pretrained=False, num_classes=1).to(self.parameter_dict["device"])
        elif self.parameter_dict["net"] == "DeepLabV3_ResNet101":
            self.model = torch.hub.load('pytorch/vision:v0.11.1', 'deeplabv3_resnet101',
                                        pretrained=False, num_classes=1).to(self.parameter_dict["device"])
        elif self.parameter_dict["net"] == "DeepLabV3_MobileNet":
            self.model = torch.hub.load('pytorch/vision:v0.11.1', 'deeplabv3_mobilenet_v3_large',
                                        pretrained=False, num_classes=1).to(self.parameter_dict["device"])

        # Ablation experiments
        elif self.parameter_dict["net"] == "UNetSharp2":
            self.model = UNetSharp2(nc=self.parameter_dict["nc"], pooling=self.parameter_dict["pooling"],
                                    dropout=self.parameter_dict["dropout"], dropout_p=self.parameter_dict["dropout_p"],
                                    block_size=self.parameter_dict["block_size"]).to(self.parameter_dict["device"])
        elif self.parameter_dict["net"] == "UNetSharp_noSP":
            self.model = UNetSharp_sinNP(nc=self.parameter_dict["nc"], pooling=self.parameter_dict["pooling"],
                                         dropout=self.parameter_dict["dropout"], dropout_p=self.parameter_dict["dropout_p"],
                                         block_size=self.parameter_dict["block_size"]).to(self.parameter_dict["device"])
            self.model.init_weights()
        else:
            # TODO: Add new models
            net = self.parameter_dict["net"]
            raise Exception(f"Model {net} not implemented/imported")

        if TORCH_VERSION.startswith("2.0"):
            self.LOG("Using torch.compile to speed up training")
            print("Using torch.compile to speed up training")
            self.model = torch.compile(self.model)

        if self.parameter_dict['multi_gpu']:
            self.model = nn.DataParallel(
                self.model, device_ids=self.parameter_dict["device_ids"])

        self.LOG("Model {} load succesfully".format(
            self.parameter_dict["net"]))

        if self.parameter_dict["optimizer"] == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.parameter_dict["learning_rate"],
                                        betas=(0.9, 0.999))
        elif self.parameter_dict["optimizer"] == 'AdaBelief':
            self.optimizer = AdaBelief(self.model.parameters(), lr=self.parameter_dict["learning_rate"],
                                       betas=(0.9, 0.999))
        else:
            opt = self.parameter_dict["optimizer"]
            raise Exception(f"Optimizer {opt} not implemented/imported")

        if self.parameter_dict["pretrained_weights"]:
            self.load_weights()
            self.LOG("Pretained weights load succesfully")

        if self.parameter_dict["tensorboard"]:
            self.tb_runs_folder = check_runs_folder(
                self.experiment_folder.split('/')[-1])
            self.writer = SummaryWriter(f"{self.tb_runs_folder}")
        else:
            self.writer = None

        self.dsc_best = -1

    @staticmethod
    def set_random_seed(random_seed):
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        imgaug.seed(random_seed)
        np.random.seed(random_seed)

    def load_weights(self, path_w=None):

        # If path is not specified it tries to load from the path in hyperparameters.yaml
        if path_w == None:
            try:
                self.model.load_state_dict(torch.load(
                    self.parameter_dict["pretrained_weights_path"],
                    map_location=self.parameter_dict["device"]
                ))

            except:
                path = self.parameter_dict["pretrained_weights_path"]
                raise Exception(
                    f"[E] Pretrained weights do not exist at {path} or they are not compatible")
        else:
            try:
                self.model.load_state_dict(torch.load(path_w))
            except:
                raise Exception(
                    f"[E] Pretrained weights do not exist at {path_w} or they are not compatible")

    def save_weights(self, best=False):

        path = os.path.join(self.experiment_folder, "weights")

        if not os.path.isdir(path):
            os.mkdir(path)

        if not best:
            torch.save(self.model.state_dict(), os.path.join(path, "last.pt"))
        else:
            torch.save(self.model.state_dict(), os.path.join(path, "best.pt"))

    def update_learning_rate(self, epoch):

        # LR scheduler
        PI = 3.14159
        n_epochs = self.parameter_dict["n_epochs"]
        initial_lr = self.parameter_dict["learning_rate"]
        lr = initial_lr * (np.cos(PI * epoch/n_epochs) + 1.) / 2

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @staticmethod
    def compute_loss(prediction, target, bce_weight=0.):

        # Binary cross entropy (BCE) loss, before hard sigmoid
        bce = F.binary_cross_entropy_with_logits(prediction, target)

        hard_sigmoid = nn.Hardsigmoid()
        prediction = hard_sigmoid(prediction)

        # DSC loss, after hard sigmoid
        dsc = torch_dice_loss(prediction, target)

        loss = bce * bce_weight + dsc * (1 - bce_weight)
        return loss

    def train_step(self):

        self.model.train()
        avg_loss = 0

        for batched_sample in tqdm(self.dataset.trainset_loader, desc="Train step     ", unit=' batches', ncols=100):

            self.optimizer.zero_grad()

            images, masks, images_orig = batched_sample["image"].to(self.parameter_dict["device"]),\
                batched_sample["mask"].to(self.parameter_dict["device"]),\
                batched_sample["image_orig"].to(self.parameter_dict["device"])

            if self.parameter_dict['adversarial_training']:
                images.requires_grad = True

            output = get_model_outputs(
                self.parameter_dict['net'], self.model, images, images_orig)

            loss = self.compute_loss(
                output, masks, bce_weight=self.parameter_dict['bce_weight'])

            avg_loss += loss.item()

            if self.parameter_dict['adversarial_training']:
                loss.backward(retain_graph=True)
                images_grad = images.grad.data
                perturbed_images = self.fgsm_attack(
                    images, images_grad, epsilon=self.parameter_dict['epsilon'])

                output = get_model_outputs(
                    self.parameter_dict['net'], self.model, images, images_orig)

                loss = self.compute_loss(output, masks)

            loss.backward()
            self.optimizer.step()

        return avg_loss / len(self.dataset.trainset_loader)

    def val_step(self):

        self.model.eval()

        with torch.no_grad():
            avg_loss = 0

            for batched_sample in tqdm(self.dataset.valset_loader, desc="Validation step", unit=' batches', ncols=100):

                images, masks, images_orig = batched_sample["image"].to(self.parameter_dict["device"]),\
                    batched_sample["mask"].to(self.parameter_dict["device"]),\
                    batched_sample["image_orig"].to(
                        self.parameter_dict["device"])

                output = get_model_outputs(
                    self.parameter_dict['net'], self.model, images, images_orig)

                loss = self.compute_loss(output, masks)

                avg_loss += loss.item()

        return avg_loss / len(self.dataset.valset_loader)

    def train(self):

        self.LOG("Starting training the model...")

        for epoch in range(1, self.parameter_dict["n_epochs"]+1):

            self.LOG(f"Starting epoch {epoch}")
            self.update_learning_rate(epoch)

            print(f"Epoch {epoch}")
            start = time.time()
            train_loss = self.train_step()
            val_loss = self.val_step()

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)

            metrics = get_evaluation_metrics(self.writer, epoch, self.dataset.valset_loader, self.model,
                                             self.parameter_dict["device"], writer=self.writer,
                                             SAVE_SEGS=False, N_EPOCHS_SAVE=10, folder=f"{self.experiment_folder}/segmentations",
                                             grayscale=self.parameter_dict["grayscale"], model=self.parameter_dict['net'])

            end = time.time()
            elapsed = end - start
            msg = f"Epoch {epoch} finished -- Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f} -- Elapsed time: {elapsed:.1f}s"
            print(msg + "\n")
            self.LOG(msg)

            self.save_weights()
            self.LOG(f"Last weights saved at epoch {epoch}")

            if metrics.dice > self.dsc_best:

                self.LOG(
                    f"New best value of DSC reach: {metrics.dice:.4f} (last: {self.dsc_best:.4f})")
                self.dsc_best = metrics.dice
                self.save_weights(best=True)
                self.LOG(f"Best weights saved at epoch {epoch}")

            self.LOG_METRICS(metrics, epoch, train_loss, val_loss)

    def test(self):

        # weights = self.parameter_dict['pretrained_weights_path']
        # weights.replace('')
        # self.load_weights('/home/imartinez/Code/experiments_arquitectura/exp11/weights/last.pt')
        # self.load_weights('/home/imartinez/Code/_experiments_/exp7/weights/last.pt')
        # self.load_weights()

        self.model.eval()

        print("[I] Evalutating the model...")

        metrics = get_evaluation_metrics(None, -1, self.dataset.testset_loader, self.model,
                                         self.parameter_dict["device"], None, COLOR=True,
                                         SAVE_SEGS=True, N_EPOCHS_SAVE=5, folder=f"{self.experiment_folder}/segmentations", filename='avg_metrics_last.txt',
                                         grayscale=self.parameter_dict["grayscale"], model=self.parameter_dict['net'])

        print(
            "\n----------------------------------------------------------------------------")
        print("EVALUTAION RESULTS ON TEST SET:")
        print("\tCCR: {:.4f}".format(metrics.CCR))
        print("\tPrecision: {:.4f}".format(metrics.precision))
        print("\tRecall: {:.4f}".format(metrics.recall))
        print("\tSensibility: {:.4f}".format(metrics.sensibility))
        print("\tSpecifity: {:.4f}".format(metrics.specifity))
        print("\tF1 score: {:.4f}".format(metrics.f1_score))
        print("\tJaccard coef: {:.4f}".format(metrics.jaccard))
        print("\tDSC coef: {:.4f}".format(metrics.dice))
        print("\tROC-AUC: {:.4f}".format(metrics.roc_auc))
        print(
            "\tPrecision-recall AUC: {:.4f}".format(metrics.precision_recall_auc))
        print("\tHausdorf error: {:.4f}".format(metrics.hausdorf_error))
        print(
            "----------------------------------------------------------------------------")
        print(f"Segmentations saved at {self.experiment_folder}/segmentations")

        self.test_per_area()

    def test_per_area(self, load=False):

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
                f.write("epoch,train_loss,val_loss,ccr,precision,recall,sensibility,specifity,f1_score," +
                        "jaccard_coef,dsc_coef,roc_auc,pr_auc,hausdorf_error\n")

        with open(file, 'a') as f:

            f.write(f"{epoch},{train_loss},{val_loss},{metrics.CCR},{metrics.precision},"
                    f"{metrics.recall},{metrics.sensibility},{metrics.specifity},{metrics.f1_score},"
                    f"{metrics.jaccard},{metrics.dice},{metrics.roc_auc},{metrics.precision_recall_auc},"
                    f"{metrics.hausdorf_error}\n")
