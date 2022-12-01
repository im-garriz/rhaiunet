import os
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
from utils import *
import numpy as np
from segmentation_metrics import get_evaluation_metrics
import random
import imgaug
from progress_logger import ProgressLogger
from rdau_net import RDAU_NET
from discriminator import Critic
from segmentation_metrics import get_evaluation_metrics


class WGAN:

    def __init__(self, root_dir, parameter_dict, dataset, writer=None):

        self.root_dir = root_dir

        if not os.path.isdir(self.root_dir):
            raise (f"[E] Path {self.root_dir} does not exist")

        self.N_EPOCHS = parameter_dict["n_epochs"]
        self.INITIAL_EPOCH = parameter_dict["initial_epoch"]
        self.GENERATOR_LEARNING_RATE = parameter_dict["generator_lr"]
        self.CRITIC_LEARNING_RATE = parameter_dict["critic_lr"]
        self.ADAPTATIVE_LEARNING_RATE = parameter_dict["adaptative_lr"]
        self.N_CRITIC = parameter_dict["n_critic"]

        self.GENERATOR_TYPE = parameter_dict["generator"]
        self.CRITIC_TYPE = parameter_dict["critic"]

        self.DEVICE = parameter_dict["device"]

        self.PRETRAINED_WEIGHTS = parameter_dict["pretained_weights"]
        self.PRETRAINED_WEIGHTS_PATH = parameter_dict["pretained_weights_path"]

        if not os.path.isdir(os.path.join(self.root_dir, self.PRETRAINED_WEIGHTS_PATH)):
            os.mkdir(os.path.join(self.root_dir, self.PRETRAINED_WEIGHTS_PATH))

        self.GENERATOR_WEIGHTS_FILE = parameter_dict["generator_weights_file"]
        self.CRITIC_WEIGHTS_FILE = parameter_dict["critic_weights_file"]

        self.OPTIMIZER = parameter_dict["optimizer"]

        self.LOG_FILE = parameter_dict["log_file"]

        self.set_random_seed(parameter_dict["random_seed"])

        self.critic_input_type = parameter_dict["critic_input_type"]#: "masked_img"  # masked_img or concatenation

        self.dataset = dataset

        self.logger = ProgressLogger(len(self.dataset.trainset_loader), len(self.dataset.valset_loader))

        if writer is not None:
            if not os.path.isdir(os.path.join(root_dir, writer)):
                os.mkdir(os.path.join(root_dir, writer))
            self.writer = SummaryWriter(os.path.join(root_dir, writer))
        else:
            self.writer = None

        self.init_wgan()

        with open(os.path.join(self.root_dir, self.LOG_FILE), 'w') as file:
            file.write("epoch,generator_loss,critic_loss,ccr,precision,recall,sensibility,specifity,f1_score,"
                       "jaccard_coef,dsc_coef,roc_auc,pr_auc,hausdorf_error\n")

    def init_wgan(self):

        if self.GENERATOR_TYPE == "RDAU-NET":
            self.generator = RDAU_NET().to(self.DEVICE)
        else:
            raise (f"[E] No generator found: {self.GENERATOR_TYPE}")

        if self.critic_input_type == "concatenation":
            critic_in_channels = 4
        elif self.critic_input_type == "masked_img":
            critic_in_channels = 3
        else:
            raise (f"[E] Not valid critic input type: {self.critic_input_type}")

        if self.CRITIC_TYPE == "Critic":
            self.critic = Critic(critic_in_channels).to(self.DEVICE)
        else:
            raise (f"[E] No critic found: {self.CRITIC_TYPE}")

        self.generator.init_weights()
        self.critic.init_weights()

        if self.PRETRAINED_WEIGHTS:
            self.load_weights()

        if self.OPTIMIZER == "Adam":
            self.optimizerG = optim.Adam(self.generator.parameters(), lr=self.GENERATOR_LEARNING_RATE,
                                         betas=(0.5, 0.999))
            self.optimizerC = optim.Adam(self.critic.parameters(), lr=self.CRITIC_LEARNING_RATE, betas=(0.5, 0.999))
        elif self.OPTIMIZER == "RMSprop":
            self.optimizerG = optim.RMSprop(self.generator.parameters(), lr=self.GENERATOR_LEARNING_RATE)
            self.optimizerC = optim.RMSprop(self.critic.parameters(), lr=self.CRITIC_LEARNING_RATE)
        else:
            raise (f"[E] Optimizer {self.OPTIMIZER} not implemented")


    def train(self):

        forward_passed_batches = 0
        for epoch in range(self.INITIAL_EPOCH, self.INITIAL_EPOCH+self.N_EPOCHS):

            self.logger.print_epoch(epoch)

            G_losses = []
            C_losses = []

            self.logger.initialize_train_bar()

            for i, batched_sample in enumerate(self.dataset.trainset_loader):

                images, masks = batched_sample["image"].to(self.DEVICE), batched_sample["mask"].to(self.DEVICE)

                loss_C = self.critic_step(images, masks)
                C_losses.append(loss_C.item())
                if self.writer is not None:
                    self.writer.add_scalar("Train loss/critic", loss_C.item(),
                                           epoch * len(self.dataset.trainset_loader) + i)
                forward_passed_batches += 1

                if forward_passed_batches == self.N_CRITIC:
                    loss_G = self.generator_step(images)
                    G_losses.append(loss_G.item())
                    if self.writer is not None:
                        self.writer.add_scalar("Train loss/generator", loss_G.item(),
                                               epoch * len(self.dataset.trainset_loader) + i)
                    forward_passed_batches = 0

                self.logger.update_bar()

            if self.writer is not None:
                self.writer.add_scalar("Per epoch train loss/generator", np.mean(np.array(G_losses)), epoch)
                self.writer.add_scalar("Per epoch train loss/discriminator", np.mean(np.array(C_losses)), epoch)

            self.validation_step(epoch, np.mean(np.array(G_losses)), np.mean(np.array(C_losses)))
            self.save_weights()

            if self.ADAPTATIVE_LEARNING_RATE:
                self.update_learning_rate(epoch)

            self.logger.finish_epoch()


    def generator_step(self, images):

        self.optimizerG.zero_grad()

        segmentations = self.generator(images)
        #images_with_segmentations = images.clone()

        if self.critic_input_type == "concatenation":
            images_with_segmentations = merge_images_with_masks(images, segmentations).to(self.DEVICE)
        elif self.critic_input_type == "masked_img":
            #segmentations = torch.autograd.Variable((segmentations > 0.5).float(), requires_grad=True)
            #images_with_segmentations = torch.autograd.Variable(images * segmentations, requires_grad=True)
            images_with_segmentations = images * segmentations

        loss_G = -torch.mean(self.critic(images_with_segmentations))

        loss_G.backward()
        self.optimizerG.step()

        return loss_G

    def critic_step(self, images, masks):

        self.optimizerC.zero_grad()

        #images_with_segmentations = images.clone()
        #images_with_masks = images.clone()

        if self.critic_input_type == "concatenation":
            images_with_masks = merge_images_with_masks(images, masks).to(self.DEVICE)
        elif self.critic_input_type == "masked_img":
            #images_with_masks = torch.autograd.Variable(images * masks, requires_grad=True)
            images_with_masks = images * masks

        segmentations = self.generator(images).detach()

        if self.critic_input_type == "concatenation":
            images_with_segmentations = merge_images_with_masks(images, segmentations).to(self.DEVICE)
        elif self.critic_input_type == "masked_img":
            #segmentations = torch.autograd.Variable((segmentations > 0.5).float(), requires_grad=True)
            #images_with_segmentations = torch.autograd.Variable(images * segmentations, requires_grad=True)
            images_with_segmentations = images * segmentations

        loss_C = -torch.mean(self.critic(images_with_masks)) + torch.mean(self.critic(images_with_segmentations))

        _gradient_penalty = self.gradient_penalty(self.critic, images_with_masks, images_with_segmentations,
                                                  10, self.DEVICE)
        loss_C += _gradient_penalty

        loss_C.backward()

        self.optimizerC.step()

        return loss_C

    def validation_step(self, epoch, generator_loss, critic_loss):

        metrics = get_evaluation_metrics(self.logger, epoch, self.dataset.valset_loader, self.generator, self.DEVICE,
                                         SAVE_SEGS=True, writer=self.writer, COLOR=True, N_EPOCHS_SAVE=10,
                                         folder=os.path.join(self.root_dir, "Samples"))

        with open(os.path.join(self.root_dir, self.LOG_FILE), 'a') as file:
            file.write(f"{epoch},{generator_loss},{critic_loss},{metrics.CCR},{metrics.precision},"
                       f"{metrics.recall},{metrics.sensibility},{metrics.specifity},{metrics.f1_score},"
                       f"{metrics.jaccard},{metrics.dice},{metrics.roc_auc},{metrics.precision_recall_auc},"
                       f"{metrics.hausdorf_error}\n")

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
    def consistency_term(real_images, discriminator, M_=0, lambda2=2.0):

        x1, x_1 = discriminator(real_images)
        x2, x_2 = discriminator(real_images)

        consistency_term = torch.square(x1-x2)
        consistency_term += 0.1 * torch.mean(torch.square(x_1-x_2), dim=[1, 2])
        consistency_term = torch.max(0.0, consistency_term - M_)

        return lambda2 * consistency_term

    @staticmethod
    def set_random_seed(random_seed):
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        imgaug.seed(random_seed)

    def load_weights(self):
        try:
            self.generator.load_state_dict(torch.load(os.path.join(self.root_dir, self.PRETRAINED_WEIGHTS_PATH,
                                                                   self.GENERATOR_WEIGHTS_FILE)))
            self.critic.load_state_dict(torch.load(os.path.join(self.root_dir, self.PRETRAINED_WEIGHTS_PATH,
                                                                self.CRITIC_WEIGHTS_FILE)))
        except:
            raise(f"[E] Pretrained weights do not exist at {os.path.join(self.root_dir, self.PRETRAINED_WEIGHTS_PATH)}")

    def save_weights(self):
        torch.save(self.generator.state_dict(), os.path.join(self.root_dir, self.PRETRAINED_WEIGHTS_PATH,
                                                             self.GENERATOR_WEIGHTS_FILE))
        torch.save(self.critic.state_dict(), os.path.join(self.root_dir, self.PRETRAINED_WEIGHTS_PATH,
                                                          self.CRITIC_WEIGHTS_FILE))

    def update_learning_rate(self, epoch):

        self.GENERATOR_LEARNING_RATE = self.GENERATOR_LEARNING_RATE * (0.1 ** (epoch // 30))
        for param_group in self.optimizerG.param_groups:
            param_group['lr'] = self.GENERATOR_LEARNING_RATE

        self.CRITIC_LEARNING_RATE = self.CRITIC_LEARNING_RATE * (0.1 ** (epoch // 30))
        for param_group in self.optimizerC.param_groups:
            param_group['lr'] = self.CRITIC_LEARNING_RATE

    def test_segmenter(self):

        metrics = get_evaluation_metrics(self.logger, -1, self.dataset.trainset_loader, self.generator, self.DEVICE,
                                         writer=None, SAVE_SEGS=True, COLOR=True, N_EPOCHS_SAVE=1,
                                         folder=os.path.join(self.root_dir, "final_inf/trainset"))

        print("----------------------------------------------------------------------------")
        print("EVALUTAION RESULTS - TRAIN SET:")
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

        metrics = get_evaluation_metrics(self.logger, -1, self.dataset.valset_loader, self.generator, self.DEVICE,
                                         writer=None, SAVE_SEGS=True, COLOR=True, N_EPOCHS_SAVE=1,
                                         folder=os.path.join(self.root_dir, "final_inf/valset"))

        print("----------------------------------------------------------------------------")
        print("EVALUTAION RESULTS - VAL SET:")
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

        metrics = get_evaluation_metrics(self.logger, -1, self.dataset.testset_loader, self.generator, self.DEVICE,
                                         writer=None, SAVE_SEGS=True, COLOR=True, N_EPOCHS_SAVE=1,
                                         folder=os.path.join(self.root_dir, "final_inf/testset"))

        print("----------------------------------------------------------------------------")
        print("EVALUTAION RESULTS - VAL SET:")
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

if __name__ == "__main__":
    pass
