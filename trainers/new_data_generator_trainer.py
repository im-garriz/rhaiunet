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

from models.segmentors.unet import UNet, UNet2
from models.segmentors.rdau_net import RDAU_NET
from models.segmentors.unetpp import UNetpp
#from self_attention_cv.transunet import TransUnet
from models.critics.critic import Critic1, Critic2, Critic3_CT_WGAN
from models.image_generators.image_generators import ImageGenerator1

from common.hyperparameters import HyperparameterReader
from common.dataset_handler import load_dataset
from common.image_transformations import load_img_transforms, UnNormalize
from common.data_augmentation import load_data_augmentation_pipes
from common.utils import torch_dice_loss, check_experiments_folder, check_runs_folder, merge_images_with_masks
from common.segmentation_metrics import get_evaluation_metrics
from common.progress_logger import ProgressBar
from common.utils import generate_output_img

class DataGenerator:

    def __init__(self, hyperparams_file):

        hyperparameter_loader = HyperparameterReader(hyperparams_file)
        self.parameter_dict = hyperparameter_loader.load_param_dict()
        
        self.set_random_seed(self.parameter_dict["random_seed"])

        if self.parameter_dict['device'] is False:
            self.parameter_dict["device"] = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
            
        if self.parameter_dict['multi_gpu']:
            self.parameter_dict["device"] = torch.device('cuda')

        transforms_dict = load_img_transforms()

        augmentation_dict = load_data_augmentation_pipes(data_aug=self.parameter_dict["data_augmentation"])

        self.parameter_dict["transforms"] = transforms_dict
        self.parameter_dict["augmentation_pipelines"] = augmentation_dict

        self.dataset = load_dataset(self.parameter_dict)

        if self.parameter_dict["critic"] == "Critic1":
            self.critic = Critic1(4).to(self.parameter_dict["device"])
            self.critic.init_weights()
        elif self.parameter_dict["critic"] == "Critic2":
            self.critic = Critic2(4).to(self.parameter_dict["device"])
            self.critic.init_weights()
        elif self.parameter_dict["critic"] == "Critic3_CT_WGAN":
            self.critic = Critic3_CT_WGAN(4).to(self.parameter_dict["device"])
            self.critic.init_weights()
        else:
            #TODO
            pass

        if self.parameter_dict['image_generator'] == 'ImageGenerator1':
            self.generator = ImageGenerator1().to(self.parameter_dict["device"])
            self.generator.init_weights()

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

        if self.parameter_dict["pretrained_image_weights_generator"]:
            self.load_weights(generator=True)
            self.LOG("Pretained weights of generator load succesfully")

        if self.parameter_dict["pretrained_weights_critic"]:
            self.load_weights(generator=False)
            self.LOG("Pretained weights of critic load succesfully")  


        self.best_loss = 999


    def load_weights(self, generator=True):
        try:
            if generator:
                self.generator.load_state_dict(torch.load(self.parameter_dict["pretrained_weights_path_image_generator"]))
            else:
                self.critic.load_state_dict(torch.load(self.parameter_dict["pretrained_weights_path_critic"]))

        except:
            if generator:
                path = self.parameter_dict["pretrained_weights_path_image_generator"]
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

        assert real_segmentations.shape == generated_segmentations.shape
        
        n_elements = real_segmentations.nelement()
        batch_size = real_segmentations.size()[0]
        colors = real_segmentations.size()[1]
        image_width = real_segmentations.size()[2]
        image_height = real_segmentations.size()[3]
        alpha = torch.rand(batch_size, 1).expand(batch_size, int(n_elements / batch_size)).contiguous()
        alpha = alpha.view(batch_size, colors, image_width, image_height).to(device)

        fake_data = generated_segmentations.view(batch_size, colors, image_width, image_height)
        #fake_data = generated_segmentations
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


    def train_step(self, forward_passed_batches):
        
        G_avg_loss = 0
        C_avg_loss = 0

        print("Train step")
        bar = ProgressBar(len(self.dataset.trainset_loader))

        for i, batched_sample in enumerate(self.dataset.trainset_loader):

            images, masks = batched_sample["image"].to(self.parameter_dict["device"]),\
                            batched_sample["mask"].to(self.parameter_dict["device"])

            #images = torch.autograd.Variable(merge_images_with_masks(images, masks), requires_grad=True).to(self.parameter_dict['device'])
            noise = torch.randn(images.shape[0], self.parameter_dict['latent_vector_size'], 1, 1).to(self.parameter_dict['device'])
            
            ## Critic step
            self.optimizerC.zero_grad()
            fake_images = self.generator(noise).detach()
            
            loss_C = -torch.mean(self.critic(images)) + torch.mean(self.critic(fake_images))
            
            _gradient_penalty = self.gradient_penalty(self.critic, images, fake_images, 10, self.parameter_dict["device"])

            loss_C += _gradient_penalty
            
            loss_C.backward()
            self.optimizerC.step()
            
            C_avg_loss += loss_C.item()
            
            forward_passed_batches += 1
            #####
            
            ## Generator step
            if forward_passed_batches == self.parameter_dict["n_critic"]:
                forward_passed_batches = 0
                
                self.optimizerG.zero_grad()
                generated_images = self.generator(noise)
                
                loss_G = -torch.mean(self.critic(generated_images))
                
                loss_G.backward()
                self.optimizerG.step()
                
                G_avg_loss += loss_G.item()
            #####

            bar.step_bar()  

        G_avg_loss /= len(self.dataset.trainset_loader)
        C_avg_loss /= len(self.dataset.trainset_loader)

        return G_avg_loss, C_avg_loss, forward_passed_batches
                    

    def train(self):

        self.init_train_scheme()

        self.LOG("Starting training the model...")
        
        with(open(os.path.join(self.experiment_folder, 'loss.csv'), 'w')) as file:
            file.write('Epoch,Generator,Critic\n')

        forward_passed_batches = 0
        for epoch in range(1, self.parameter_dict["n_epochs"]+1):

            self.LOG(f"Starting epoch {epoch}")

            start = time.time()
            train_gen_loss, train_crit_loss, forward_passed_batches = self.train_step(forward_passed_batches)
            end = time.time()
            elapsed = end - start
            
            with(open(os.path.join(self.experiment_folder, 'loss.csv'), 'a')) as file:
                file.write(f'{epoch},{train_gen_loss},{train_crit_loss}\n')

            if self.parameter_dict['mlflow']:
                mlflow.log_metric("train_gen_loss", train_gen_loss)
                mlflow.log_metric("train_crit_loss", train_crit_loss)

            msg = f"Epoch {epoch} finished -- Generator train loss: {train_gen_loss:.4f} - " +\
                  f"Critic train loss: {train_crit_loss:.4f} -- Elapsed time: {elapsed:.1f}s"
            print(msg + "\n")
            self.LOG(msg)

            if self.writer is not None:
                self.writer.add_scalar("Train loss/generator", train_gen_loss, epoch)
                if train_crit_loss < 100:
                    self.writer.add_scalar("Train loss/critic", train_crit_loss, epoch)

            
            self.save_weights()
            self.LOG(f"Last weights saved at epoch {epoch}")
            
            if epoch % 50 == 0:
                
                SAVE_FOLDER = f"generated/epoch {epoch}"
                DEVICE = self.parameter_dict['device']
                
                if not os.path.isdir('generated'):
                        os.mkdir('generated')
                
                noise = torch.randn(25, 100, 1, 1).to(DEVICE)
                output = self.generator(noise)
                
                n_images = output.shape[0]
    
                trans = transforms.ToPILImage()
        
                for i in range(n_images):
                
                    mask = output[i, 3, :, :].detach().to('cpu')
                    image = output[i, :3, :, :].detach().to('cpu')

                    image = trans(image)
                    mask = trans(mask)

                    np_image = np.array(image)
                    np_mask = np.array(mask)

                    np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)

                    save_img = np.hstack([np_image, np_mask])

                    if not os.path.isdir(SAVE_FOLDER):
                        os.mkdir(SAVE_FOLDER)

                    cv2.imwrite(os.path.join(SAVE_FOLDER, f"sample_{i}.png"), save_img)
                


    def init_train_scheme(self):

        self.experiment_folder = check_experiments_folder()

        if self.parameter_dict['mlflow']:
            import mlflow
            from mlflow.tracking.fluent import set_experiment
            
            mlflow.set_tag('Exp', self.experiment_folder)
            mlflow.log_artifacts(self.experiment_folder, artifact_path="log")

            
        self.LOG("Launching DataGenerator...")
        self.LOG("Hyperparameters succesfully read from {hyperparams_file}:")
        for key, val in self.parameter_dict.items():
            self.LOG(f"\t{key}: {val}")
            
            if self.parameter_dict['mlflow']:
                mlflow.log_param(key, val)

        if self.parameter_dict["tensorboard"]:
            self.tb_runs_folder = check_runs_folder(self.experiment_folder.split('/')[-1])
            self.writer = SummaryWriter(f"{self.tb_runs_folder}")
        else:
            self.writer = None


    def LOG(self, msg):

        file = os.path.join(self.experiment_folder, "log.txt")

        if self.parameter_dict['mlflow']:
            mlflow.log_artifacts(self.experiment_folder, artifact_path="log")

        if not os.path.isfile(file):
            with open(file, 'w') as f:
                pass

        with open(file, 'a') as f:

            timestamp = str(datetime.now()).split('.')[0]
            f.write(f"{timestamp}: {msg}\n")



    @staticmethod
    def set_random_seed(random_seed):
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        imgaug.seed(random_seed)
        np.random.seed(random_seed)
