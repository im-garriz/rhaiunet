import time
import os
import random
import imgaug
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.segmentors.unet import UNet, UNet2
from models.segmentors.rdau_net import RDAU_NET
from models.segmentors.unetpp import UNetpp
from self_attention_cv.transunet import TransUnet

from models.critics.critic import Critic1, Critic2, Critic3_CT_WGAN

from common.hyperparameters import HyperparameterReader
from common.dataset_handler import load_dataset
from common.image_transformations import load_img_transforms, UnNormalize
from common.data_augmentation import load_data_augmentation_pipes
from common.utils import check_experiments_folder, check_runs_folder, merge_images_with_masks
from common.segmentation_metrics import get_evaluation_metrics
from common.progress_logger import ProgressBar


class WGanTrainer:

    def __init__(self, hyperparams_file):
        
        self.experiment_folder = check_experiments_folder()

        hyperparameter_loader = HyperparameterReader(hyperparams_file)
        self.parameter_dict = hyperparameter_loader.load_param_dict()

        self.LOG("Launching WganTrainer...")
        self.LOG("Hyperparameters succesfully read from {hyperparams_file}:")
        for key, val in self.parameter_dict.items():
            self.LOG(f"\t{key}: {val}")

        self.set_random_seed(self.parameter_dict["random_seed"])

        #self.parameter_dict["device"] = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        #self.LOG("Found device: {}".format(self.parameter_dict["device"]))

        transforms_dict = load_img_transforms()
        self.LOG("Transforms dict load succesfully")

        augmentation_dict = load_data_augmentation_pipes(data_aug=self.parameter_dict["data_augmentation"])
        self.LOG("Data augmentation dict load succesfully")

        self.parameter_dict["transforms"] = transforms_dict
        self.parameter_dict["augmentation_pipelines"] = augmentation_dict

        self.dataset = load_dataset(self.parameter_dict)
        self.LOG("Dataset load succesfully")

        if self.parameter_dict["net"] == "UNet":
            self.generator = UNet(1, 1, bilinear=self.parameter_dict["bilinear"]).to(self.parameter_dict["device"])
            self.generator.init_weights()
        elif self.parameter_dict["net"] == "RDAUNet":
            self.generator = RDAU_NET().to(self.parameter_dict["device"])
            self.generator.init_weights()
        elif self.parameter_dict["net"] == "UNet2":
            self.generator = UNet2(1, 1).to(self.parameter_dict["device"])
            self.generator.init_weights()
        elif self.parameter_dict["net"] == "UNetpp":
            self.generator = UNetpp(1).to(self.parameter_dict["device"])
            self.generator.init_weights()
        else:
            #TODO
            pass

        if self.parameter_dict["critic"] == "Critic1":
            self.critic = Critic1(2).to(self.parameter_dict["device"])
            self.critic.init_weights()
        elif self.parameter_dict["critic"] == "Critic2":
            self.critic = Critic2(2).to(self.parameter_dict["device"])
            self.critic.init_weights()
        elif self.parameter_dict["critic"] == "Critic3_CT_WGAN":
            self.critic = Critic3_CT_WGAN(2).to(self.parameter_dict["device"])
            self.critic.init_weights()
        else:
            #TODO
            pass

        self.LOG("Generator {} load succesfully".format(self.parameter_dict["net"]))
        self.LOG("Critic {} load succesfully".format(self.parameter_dict["critic"]))


        if self.parameter_dict["optimizer"] == "Adam":
            self.optimizerG = optim.Adam(self.generator.parameters(), lr=self.parameter_dict["generator_learning_rate"],
                                        betas=(self.parameter_dict["adam_beta1"], self.parameter_dict["adam_beta2"]))
            self.optimizerC = optim.Adam(self.critic.parameters(), lr=self.parameter_dict["critic_learning_rate"],
                                        betas=(self.parameter_dict["adam_beta1"], self.parameter_dict["adam_beta2"]))

        elif self.parameter_dict["optimizer"] == "RMSProp":

            self.optimizerG = optim.RMSprop(self.generator.parameters(),
                                            lr=self.parameter_dict["generator_learning_rate"])
            self.optimizerC = optim.RMSprop(self.critic.parameters(),
                                            lr=self.parameter_dict["critic_learning_rate"])
        else:
            # TODO
            pass

        if self.parameter_dict["pretrained_weights_generator"]:
            self.load_weights(generator=True)
            self.LOG("Pretained weights of generator load succesfully")

        if self.parameter_dict["pretrained_weights_critic"]:
            self.load_weights(generator=False)
            self.LOG("Pretained weights of critic load succesfully")  

        if self.parameter_dict["tensorboard"]:
            self.tb_runs_folder = check_runs_folder(self.experiment_folder.split('/')[-1])
            self.writer = SummaryWriter(f"{self.tb_runs_folder}")
        else:
            self.writer = None

        #self.un_normalizer = UnNormalize(mean=0.485, std=0.225)

        self.dsc_best =  -1


    @staticmethod
    def set_random_seed(random_seed):
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        imgaug.seed(random_seed)
        np.random.seed(random_seed)


    def load_weights(self, generator=True, path_w=None):
        
        try:
            if generator:
                if path_w == None:
                    self.generator.load_state_dict(torch.load(self.parameter_dict["pretrained_weights_path_generator"]))
                else:
                    self.generator.load_state_dict(torch.load(os.path.join('/home/imartinez/Code', path_w)))
            else:
                self.critic.load_state_dict(torch.load(self.parameter_dict["pretrained_weights_path_critic"]))

        except:
            if generator:
                path = self.parameter_dict["pretrained_weights_path_generator"]
            else:
                path = self.parameter_dict["pretrained_weights_path_critic"]

            raise Exception(f"[E] Pretrained weights do not exist at {path} or they are not compatible")


    def save_weights(self, best=False):

        path = os.path.join(self.experiment_folder, "weights")

        if not os.path.isdir(path):
            os.mkdir(path)

        if not best:
            torch.save(self.generator.state_dict(), os.path.join(path, "generator_last.pt"))  
            torch.save(self.critic.state_dict(), os.path.join(path, "critic_last.pt"))    
        else:
            torch.save(self.generator.state_dict(), os.path.join(path, "generator_best.pt"))  
            torch.save(self.critic.state_dict(), os.path.join(path, "critic_best.pt")) 

    
    @staticmethod
    def gradient_penalty(critic, real_segmentations, generated_segmentations, penalty, device):

        n_elements = real_segmentations.nelement()
        batch_size = real_segmentations.size()[0]
        colors = real_segmentations.size()[1]
        image_width = real_segmentations.size()[2]
        image_height = real_segmentations.size()[3]
        alpha = torch.rand(batch_size, 1).expand(batch_size, int(n_elements / batch_size)).contiguous()
        alpha = alpha.view(batch_size, colors, image_width, image_height).to(device)

        fake_data = generated_segmentations.view(batch_size, colors, image_width, image_height)
        interpolates = alpha * generated_segmentations.detach() + ((1 - alpha) * fake_data.detach())

        interpolates = interpolates.to(device)
        interpolates.requires_grad_(True)
        critic_interpolates = critic(interpolates)

        gradients = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(critic_interpolates.size()).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty

        return gradient_penalty

    @staticmethod
    def consistency_term(critic, real_segmentations, M, penalty, device):
        
        dw_x1, dw_x1_i = critic(real_segmentations, dropout=0.5, intermediate_output=True) # Perturb the input by applying dropout to hidden layers.
        dw_x2, dw_x2_i = critic(real_segmentations, dropout=0.5, intermediate_output=True)

        # Using l2 norm as the distance metric d, referring to the official code (paper ambiguous on d).
        second_to_last_reg = ((dw_x1_i-dw_x2_i) ** 2).mean(dim=1).mean(dim=0)
        last = ((dw_x1-dw_x2) ** 2).mean(dim=1)

        d_wct_loss = last + 0.1 * second_to_last_reg - M

        d_wct_loss, _ = torch.max(d_wct_loss, 0)

        return penalty * d_wct_loss


    def generator_step(self, images):

        self.optimizerG.zero_grad()

        hard_sigmoid = nn.Hardsigmoid()
        segmentations = hard_sigmoid(self.generator(images))

        images_with_segmentations = torch.autograd.Variable(merge_images_with_masks(images, segmentations),
                                                            requires_grad=True).to(self.parameter_dict['device'])

        loss_G = -torch.mean(self.critic(images_with_segmentations))

        loss_G.backward()
        self.optimizerG.step()

        return loss_G


    def critic_step(self, images, masks, images_with_masks):

        self.optimizerC.zero_grad()

        hard_sigmoid = nn.Hardsigmoid()
        segmentations = hard_sigmoid(self.generator(images).detach())

        images_with_segmentations = torch.autograd.Variable(merge_images_with_masks(images, segmentations),
                                                            requires_grad=True).to(self.parameter_dict['device'])
        

        loss_C = -torch.mean(self.critic(images_with_masks)) + torch.mean(self.critic(images_with_segmentations))

        _gradient_penalty = self.gradient_penalty(self.critic, images_with_masks, images_with_segmentations,
                                                  10, self.parameter_dict["device"])

        loss_C += _gradient_penalty

        if self.parameter_dict["consistency_term"]:
            _consistency_term = self.consistency_term(self.critic, images_with_masks,
                                                    M=self.parameter_dict["M"], penalty=2,
                                                    device=self.parameter_dict["device"])

            loss_C += _consistency_term

        loss_C.backward()

        self.optimizerC.step()

        return loss_C


    def train_step(self, forward_passed_batches):
        
        G_avg_loss = 0
        C_avg_loss = 0

        print("Train step")
        bar = ProgressBar(len(self.dataset.trainset_loader))
        
        for i, batched_sample in enumerate(self.dataset.trainset_loader):

            images, masks = batched_sample["image"].to(self.parameter_dict["device"]),\
                            batched_sample["mask"].to(self.parameter_dict["device"])
            
            images_c_masks = batched_sample['image_p_mask'].to(self.parameter_dict["device"])

            loss_C = self.critic_step(images, masks, images_c_masks)
            C_avg_loss += loss_C.item()
            forward_passed_batches += 1

            if forward_passed_batches == self.parameter_dict["n_critic"]:
                loss_G = self.generator_step(images)
                G_avg_loss += loss_G.item()

                forward_passed_batches = 0


            bar.step_bar()

        G_avg_loss /= len(self.dataset.trainset_loader)
        C_avg_loss /= len(self.dataset.trainset_loader)

        return G_avg_loss, C_avg_loss, forward_passed_batches


    def train(self):

        self.LOG("Starting training the model...")

        forward_passed_batches = 0
        for epoch in range(1, self.parameter_dict["n_epochs"]+1):

            self.LOG(f"Starting epoch {epoch}")

            start = time.time()
            train_gen_loss, train_crit_loss, forward_passed_batches = self.train_step(forward_passed_batches)
            end = time.time()
            elapsed = end - start

            msg = f"Epoch {epoch} finished -- Generator train loss: {train_gen_loss:.4f} - " +\
                  f"Critic train loss: {train_crit_loss:.4f} -- Elapsed time: {elapsed:.1f}s"
            print(msg + "\n")
            self.LOG(msg)

            if self.writer is not None:
                self.writer.add_scalar("Train loss/generator", train_gen_loss, epoch)
                if train_crit_loss < 100:
                    self.writer.add_scalar("Train loss/critic", train_crit_loss, epoch)

            metrics = get_evaluation_metrics(self.writer, epoch, self.dataset.valset_loader, self.generator,
                                    self.parameter_dict["device"], writer=self.writer,
                                    SAVE_SEGS=False, N_EPOCHS_SAVE=20, folder=f"{self.experiment_folder}/segmentations")

            self.save_weights()
            self.LOG(f"Last weights saved at epoch {epoch}")

            if metrics.dice > self.dsc_best:

                self.LOG(f"New best value of DSC reach: {metrics.dice:.4f} (last: {self.dsc_best:.4f})")
                self.dsc_best = metrics.dice
                self.save_weights(best=True)
                self.LOG(f"Best weights saved at epoch {epoch}")

            self.LOG_METRICS(metrics, epoch, train_gen_loss, train_crit_loss)


    def test(self):

        #self.load_weights()
        self.generator.eval()

        print("[I] Evalutating the model...")

        metrics = get_evaluation_metrics(None, -1, self.dataset.testset_loader, self.generator,
                                    self.parameter_dict["device"], None, COLOR=True,
                                    SAVE_SEGS=False, N_EPOCHS_SAVE=5, folder=f"{self.experiment_folder}/segmentations", filename='avg_metrics_last.txt')

        print("\n----------------------------------------------------------------------------")
        print("EVALUTAION RESULTS (LAST):")
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
        
        
        best_w = os.path.join(self.experiment_folder, "weights", 'generator_best.pt')
        self.load_weights(best_w)
        
        print("[I] Evalutating the model...")

        metrics = get_evaluation_metrics(None, -1, self.dataset.testset_loader, self.generator,
                                    self.parameter_dict["device"], None, COLOR=True,
                                    SAVE_SEGS=True, N_EPOCHS_SAVE=5, folder=f"{self.experiment_folder}/segmentations", filename='avg_metrics_best.txt')

        print("\n----------------------------------------------------------------------------")
        print("EVALUTAION RESULTS (BEST):")
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


    def LOG(self, msg):

        file = os.path.join(self.experiment_folder, "log.txt")

        if not os.path.isfile(file):
            with open(file, 'w') as f:
                pass

        with open(file, 'a') as f:

            timestamp = str(datetime.now()).split('.')[0]
            f.write(f"{timestamp}: {msg}\n")


    def LOG_METRICS(self, metrics, epoch, generator_train_loss, critic_train_loss):

        file = os.path.join(self.experiment_folder, "metrics.csv")

        if not os.path.isfile(file):
            with open(file, 'w') as f:
                f.write("epoch,generator_train_loss,critic_train_loss,"+
                        "ccr,precision,recall,sensibility,specifity,f1_score,"+
                        "jaccard_coef,dsc_coef,roc_auc,pr_auc,hausdorf_error\n")

        with open(file, 'a') as f:

            f.write(f"{epoch},{generator_train_loss},"
                    f"{critic_train_loss},{metrics.CCR},{metrics.precision},"
                    f"{metrics.recall},{metrics.sensibility},{metrics.specifity},{metrics.f1_score},"
                    f"{metrics.jaccard},{metrics.dice},{metrics.roc_auc},{metrics.precision_recall_auc},"
                    f"{metrics.hausdorf_error}\n")


if __name__ == '__main__':

    wgan_trainer = WGanTrainer("hyperparameters.yaml")