import os

import torch
from torchvision.utils import save_image, make_grid
import torch.optim.lr_scheduler as lr_scheduler

from model import Generator
from model import Discriminator

from dataloader import data_loader
from utils import cycle
from torch.nn import DataParallel

from torch.utils.tensorboard import SummaryWriter

dev = 'cpu'
if torch.cuda.is_available():
    dev = 'cuda'


class Solver():
    def __init__(self, config, channel_list):
        # Config - Model
        self.z_dim = config.z_dim
        self.channel_list = channel_list

        # Config - Training
        self.batch_size = config.batch_size
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.decay_ratio = config.decay_ratio
        self.decay_iter = config.decay_iter
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.n_critic = config.n_critic
        self.lambda_gp = config.lambda_gp
        self.max_iter = config.max_iter

        # Config - Test
        self.fixed_z = torch.rand(128, config.z_dim, 1, 1).to(dev)

        # Config - Path
        self.data_root = config.data_root
        self.log_root = config.log_root
        self.model_root = config.model_root
        self.sample_root = config.sample_root
        self.result_root = config.result_root

        # Config - Miscellanceous
        self.print_loss_iter = config.print_loss_iter
        self.save_image_iter = config.save_image_iter
        self.save_parameter_iter = config.save_parameter_iter
        self.save_log_iter = config.save_log_iter

        self.writer = SummaryWriter(self.log_root)

    def build_model(self):
        self.G = Generator(channel_list=self.channel_list)
        self.G_ema = Generator(channel_list=self.channel_list)
        self.D = Discriminator(channel_list=self.channel_list)

        self.G = DataParallel(self.G).to(dev)
        self.G_ema = DataParallel(self.G_ema).to(dev)
        self.D = DataParallel(self.D).to(dev)

        self.g_optimizer = torch.optim.Adam(params=self.G.parameters(), lr=self.g_lr, betas=[self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(params=self.D.parameters(), lr=self.d_lr, betas=[self.beta1, self.beta2])

        self.g_scheduler = lr_scheduler.StepLR(self.g_optimizer, step_size=self.decay_iter, gamma=self.decay_ratio)
        self.d_scheduler = lr_scheduler.StepLR(self.d_optimizer, step_size=self.decay_iter, gamma=self.decay_ratio)

        print("Print model G, D")
        print(self.G)
        print(self.D)

    def load_model(self, pkl_path, channel_list):
        ckpt = torch.load(pkl_path)

        self.G = Generator(channel_list=channel_list)
        self.G_ema = Generator(channel_list=channel_list)
        self.D = Discriminator(channel_list=channel_list)

        self.G = DataParallel(self.G).to(dev)
        self.G_ema = DataParallel(self.G_ema).to(dev)
        self.D = DataParallel(self.D).to(dev)

        self.G.load_state_dict(ckpt["G"])
        self.G_ema.load_state_dict(ckpt["G_ema"])
        self.D.load_state_dict(ckpt["D"])

    def save_model(self, iters, step):
        file_name = 'ckpt_%d_%d.pkl' % ((2*(2**(step+1)), iters))
        ckpt_path = os.path.join(self.model_root, file_name)
        ckpt = {
            'G': self.G_ema.state_dict(),
            'D': self.D.state_dict()
        }
        torch.save(ckpt, ckpt_path)

    def save_img(self, iters, fixed_z, step):
        img_path = os.path.join(self.sample_root, "%d_%d.png" % (2*(2**(step+1)), iters))
        with torch.no_grad():
            generated_imgs = self.G_ema(fixed_z[:self.batch_size].to(dev), step, 1)
            save_image(make_grid(generated_imgs.cpu()/2+1/2, nrow=4, padding=2), img_path)

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def lr_update(self):
        self.g_scheduler.step()
        self.d_scheduler.step()

    def set_phase(self, mode="train"):
        if mode == "train":
            self.G.train()
            self.G_ema.train()
            self.D.train()
        elif mode == "test":
            self.G.eval()
            self.G_ema.eval()
            self.D.eval()

    def exponential_moving_average(self, beta=0.999):
        with torch.no_grad():
            G_param_dict = dict(self.G.named_parameters())
            for name, g_ema_param in self.G_ema.named_parameters():
                g_param = G_param_dict[name]
                g_ema_param.copy_(beta * g_ema_param + (1. - beta) * g_param)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(dev)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def train(self):
        # build model
        self.build_model()

        for step in range(len(self.channel_list)):
            if step > 4:
                self.batch_size = self.batch_size // 2
            loader = data_loader(self.data_root, self.batch_size, img_size=2 * (2 ** (step + 1)))
            loader = iter(cycle(loader))

            if step == 0 or step == 1 or step == 2:
                self.max_iter = 20000
            elif step == 3 or step == 4 or step == 5:
                self.max_iter = 50000
            else:
                self.max_iter = 100000

            alpha = 0.0

            for iters in range(self.max_iter+1):
                real_img = next(loader)
                real_img = real_img.to(dev)

                # ===============================================================#
                #                    1. Train the discriminator                  #
                # ===============================================================#
                self.set_phase(mode="train")
                self.reset_grad()

                # Compute loss with real images.
                d_real_out = self.D(real_img, step, alpha)
                d_loss_real = - d_real_out.mean()

                # Compute loss with face images.
                z = torch.rand(self.batch_size, self.z_dim, 1, 1).to(dev)
                fake_img = self.G(z, step, alpha)
                d_fake_out = self.D(fake_img.detach(), step, alpha)
                d_loss_fake = d_fake_out.mean()

                # Compute loss for gradient penalty.
                beta = torch.rand(self.batch_size, 1, 1, 1).to(dev)
                x_hat = (beta * real_img.data + (1 - beta) * fake_img.data).requires_grad_(True)
                d_x_hat_out = self.D(x_hat, step, alpha)
                d_loss_gp = self.gradient_penalty(d_x_hat_out, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
                d_loss.backward()
                self.d_optimizer.step()

                # ===============================================================#
                #                      2. Train the Generator                    #
                # ===============================================================#

                if (iters + 1) % self.n_critic == 0:
                    self.reset_grad()

                    # Compute loss with fake images.
                    fake_img = self.G(z, step, alpha)
                    d_fake_out = self.D(fake_img, step, alpha)
                    g_loss = - d_fake_out.mean()

                    # Backward and optimize.
                    g_loss.backward()
                    self.g_optimizer.step()

                # ===============================================================#
                #                   3. Save parameters and images                #
                # ===============================================================#
                # self.lr_update()
                torch.cuda.synchronize()
                alpha += 1 / (self.max_iter // 2)
                self.set_phase(mode="test")
                self.exponential_moving_average()

                # Print total loss
                if iters % self.print_loss_iter == 0:
                    print("Step : [%d/%d], Iter : [%d/%d], D_loss : [%.3f, %.3f, %.3f., %.3f], G_loss : %.3f" % (
                        step, len(self.channel_list)-1, iters, self.max_iter, d_loss.item(), d_loss_real.item(),
                        d_loss_fake.item(), d_loss_gp.item(), g_loss.item()
                    ))

                # Save generated images.
                if iters % self.save_image_iter == 0:
                    self.save_img(iters, self.fixed_z, step)

                # Save the G and D parameters.
                if iters % self.save_parameter_iter == 0:
                    self.save_model(iters, step)

                # Save the logs on the tensorboard.
                if iters % self.save_log_iter == 0:
                    self.writer.add_scalar('g_loss/g_loss', g_loss.item(), iters)
                    self.writer.add_scalar('d_loss/d_loss_total', d_loss.item(), iters)
                    self.writer.add_scalar('d_loss/d_loss_real', d_loss_real.item(), iters)
                    self.writer.add_scalar('d_loss/d_loss_fake', d_loss_fake.item(), iters)
                    self.writer.add_scalar('d_loss/d_loss_gp', d_loss_gp.item(), iters)

