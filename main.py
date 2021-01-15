from torch import optim, autograd
from torchvision import transforms as T
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model.PGGAN import *
from data.dataset import CelebA
from utils import EMA
from config import config


class PGGAN(object):
    def __init__(self, generator: Generator, discriminator: Discriminator, dataset: CelebA, n_critic=1,
                 lr=0.001, lr_decay=0, beta_0=0, beta_1=0.99, switch_mode_number=800000, switch_number_increase=0,
                 use_ema=True, ema_mu=0.999, use_cuda=True):
        self.G = generator
        self.D = discriminator
        assert generator.R == discriminator.R
        self.dataset = dataset
        self.n_critic = n_critic
        # Discriminator “看” 过的真实图片达到switch_mode_number时,进行模式的切换
        self.switch_mode_number = switch_mode_number
        self.switch_number_increase = switch_number_increase
        self.use_ema = use_ema
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.R = generator.R

        # self.batchsizes = {2: 128, 3: 128, 4: 128, 5: 64, 6: 64, 7: 32, 8: 8}
        self.batch_sizes = {2: 16, 3: 16, 4: 16, 5: 16, 6: 16, 7: 16, 8: 8}

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

        if self.use_ema:
            self.ema = EMA(ema_mu)
            for name, param in self.G.named_parameters():
                if param.requires_grad:
                    self.ema.register(name, param.data)

        self.lr = lr
        self.lr_decay = lr_decay
        self.D_optim = optim.Adam(self.D.parameters(), lr=lr, betas=(beta_0, beta_1), eps=1e-8)
        self.G_optim = optim.Adam(self.G.parameters(), lr=lr, betas=(beta_0, beta_1), eps=1e-8)

        self.level = 2
        self.mode = 'stabilize'
        self.current_batch_size = self.batch_sizes.get(self.level)
        # 记录当前阶段Discriminator “看” 过的真实图片数
        self.passed_real_images_num = 0
        # 'transition'模式下的fade in系数
        self.fade_in_alpha = 0
        # 'transition'模式下,每次迭代alpha增加的步长
        self.alpha_step = 1 / (self.switch_mode_number / self.current_batch_size)

        self.is_finished = 0  # 训练结束标识

        self.log_list = []  # 记录训练过程

    def update_lr(self):
        if self.lr_decay == 0:
            return
        self.lr *= self.lr_decay
        for param_group in self.D_optim.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.G_optim.param_groups:
            param_group['lr'] = self.lr

    def update_state(self):
        self.passed_real_images_num = 0
        torch.cuda.empty_cache()
        if self.mode == 'stabilize':
            if self.level == self.R:
                # level达到最大值且状态为stabilize时,终止训练
                self.is_finished = 1
                return
            self.mode = 'transition'
            self.level += 1
            self.current_batch_size = self.batch_sizes.get(self.level)
            self.fade_in_alpha = 0
            self.alpha_step = 1 / (self.switch_mode_number / self.current_batch_size)
            self.switch_mode_number += self.switch_number_increase
            self.update_lr()
        else:
            self.mode = 'stabilize'

    def update_G_weights_by_ema(self):
        """
        此函数通过EMA(Exponential Moving Average)对Generator的参数进行更新
        """
        for name, param in self.G.named_parameters():
            if param.requires_grad:
                param.data = self.ema(name, param.data)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        batch_size = real_samples.shape[0]
        alpha = torch.rand(size=(batch_size, 1, 1, 1))
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = self.D(interpolates, level=self.level, mode=self.mode, alpha=self.fade_in_alpha)
        grad = autograd.grad(outputs=d_interpolates,
                             inputs=interpolates,
                             grad_outputs=torch.ones_like(d_interpolates),
                             create_graph=True, retain_graph=True)[0]
        gradient_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_G(self):
        self.G_optim.zero_grad()
        noise = torch.randn(size=(self.current_batch_size, self.G.latent_dim))
        if self.use_cuda:
            noise = noise.cuda()
        generated_images = self.G(noise, level=self.level, mode=self.mode, alpha=self.fade_in_alpha)
        score = - self.D(generated_images, level=self.level, mode=self.mode, alpha=self.fade_in_alpha).mean()
        score.backward()
        self.G_optim.step()
        self.update_G_weights_by_ema()

    def train_D(self):
        self.D_optim.zero_grad()
        noise = torch.randn(size=(self.current_batch_size, self.G.latent_dim))
        real_images = self.dataset(self.current_batch_size, self.level)
        if self.use_cuda:
            noise, real_images = noise.cuda(), real_images.cuda()
        fake_images = self.G(noise, level=self.level, mode=self.mode, alpha=self.fade_in_alpha)
        real_score = self.D(real_images, level=self.level, mode=self.mode, alpha=self.fade_in_alpha).mean()
        fake_score = self.D(fake_images, level=self.level, mode=self.mode, alpha=self.fade_in_alpha).mean()
        gradient_penalty = self.compute_gradient_penalty(real_images, fake_images)
        epsilon_penalty = 1e-3 * torch.mean(real_score ** 2)  # 防止正例得分离0过远
        w_dist = fake_score - real_score  # 模型评价指标
        loss = w_dist + 10 * gradient_penalty + epsilon_penalty
        loss.backward()
        self.D_optim.step()
        self.passed_real_images_num += self.current_batch_size
        w_dist = w_dist.abs().item()
        print(f'\rLevel: {self.level} | Mode: {self.mode} | W-Distance: {w_dist} | Image Passed: {self.passed_real_images_num}',
              end='', file=sys.stdout, flush=True)
        self.log_list.append(w_dist)

    def plot_stat_curve(self):
        plt.plot(self.log_list)
        plt.title(f'Level_{self.level}_W_distance')
        plt.savefig(f'plots/Level_{self.level}.jpg')
        plt.close()
        self.log_list.clear()

    def train(self):
        while 1:
            for _ in range(self.n_critic):
                self.train_D()
                if self.mode == 'transition':
                    self.fade_in_alpha += self.alpha_step
                if self.passed_real_images_num >= self.switch_mode_number:
                    if self.mode == 'stabilize':
                        self.plot_stat_curve()
                        torch.save(self.G.state_dict(), f'state_dict/G_{2 ** self.level}.pkl')
                    self.update_state()
            self.train_G()
            if self.is_finished:
                break


if __name__ == '__main__':
    g = Generator(resolution=config.resolution,
                  latent_dim=config.latent_dim,
                  max_feature_map=config.max_feature_map,
                  normalize_latent=config.normalize_latent,
                  use_pixelnorm=config.use_pixelnorm,
                  use_weightscale=config.use_weightscale,
                  use_leakyrelu=config.use_leaky,
                  negative_slope=config.negative_slope,
                  tanh_at_end=config.tanh_at_end)
    d = Discriminator(resolution=config.resolution,
                      max_feature_map=config.max_feature_map,
                      use_leakyrelu=config.use_leaky,
                      negative_slope=config.negative_slope,
                      minibatch_stat_concat=config.minibatch_stat_concat,
                      use_weightscale=config.use_weightscale,
                      use_gdrop=config.use_gdrop,
                      sigmoid_at_end=config.sigmoid_at_end)
    dataset = CelebA(data_root=config.data_root,
                     transform=T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])]),
                     max_resolution=config.resolution)
    pggan = PGGAN(g, d, dataset, n_critic=config.n_critic, switch_mode_number=config.switch_mode_number,
                  switch_number_increase=config.switch_number_increase, lr=config.lr, lr_decay=config.lr_decay,
                  beta_0=config.beta0, beta_1=config.beta1, use_ema=config.use_ema, ema_mu=config.ema_mu,
                  use_cuda=config.use_cuda)
    pggan.train()
