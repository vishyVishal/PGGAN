from torch import optim, autograd
from torchvision import transforms as T
import sys
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model.PGGAN import *
from data.dataset import ImageDataset
from utils import EMA
from config import config
from swd.swd import swd


class PGGAN(object):
    def __init__(self, generator: Generator, discriminator: Discriminator, dataset: ImageDataset, n_critic=1,
                 lr=0.001, lr_decay=0, beta_0=0, beta_1=0.99, switch_mode_number=800000, switch_number_increase=0,
                 use_ema=True, ema_mu=0.999, use_cuda=True, compute_swd_every=10000, **kwargs):
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

        self.batch_sizes = {2: 128, 3: 128, 4: 128, 5: 64, 6: 32, 7: 16, 8: 8}
        # self.batch_sizes = {2: 16, 3: 16, 4: 16, 5: 16, 6: 16, 7: 16, 8: 8}

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

        self.d_loss = []  # 记录训练过程中的D Loss
        self.swd = []  # 记录训练过程中的SWD(Sliced Wasserstein Distance)

        self.compute_swd_every = compute_swd_every  # 每过多少张图片记录一次swd
        self.record_number = 0

    def update_lr(self):
        if self.lr_decay == 0:
            return
        self.lr *= self.lr_decay
        for param_group in self.D_optim.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.G_optim.param_groups:
            param_group['lr'] = self.lr

    @torch.no_grad()
    def sample_swd(self):
        if self.level <= 3:
            return
        d = 0
        for _ in range(25):
            noise = torch.randn(4, self.G.latent_dim)
            if self.use_cuda:
                noise = noise.cuda()
            img1 = self.dataset(4, level=self.level)
            img2 = self.G(noise, level=self.level).cpu()
            d += swd(img1, img2, device='cuda').item()
        return d / 25000

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
            self.switch_mode_number += self.switch_number_increase
            self.alpha_step = 1 / (self.switch_mode_number / self.current_batch_size)
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
        if self.use_ema:
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
        loss = fake_score - real_score + 10 * gradient_penalty + epsilon_penalty
        loss.backward()
        self.D_optim.step()
        self.passed_real_images_num += self.current_batch_size
        print(f'\rLevel: {self.level} | Mode: {self.mode} | D-Loss: {loss.item()} | '
              f'Image Passed: {self.passed_real_images_num}/{self.switch_mode_number}',
              end='', file=sys.stdout, flush=True)
        self.d_loss.append(loss.item())

    def plot_stat_curve(self):
        plt.plot(self.d_loss)
        plt.title(f'Level_{self.level}_D_Loss')
        plt.savefig(f'plots/Level_{self.level}_D_Loss.jpg')
        plt.close()
        self.d_loss.clear()
        if not self.swd:
            return
        plt.plot(self.swd)
        plt.title(f'Level_{self.level}_SWD')
        plt.savefig(f'plots/Level_{self.level}_SWD.jpg')
        plt.close()
        self.swd.clear()

    def train(self):
        while 1:
            for _ in range(self.n_critic):
                self.train_D()
                if self.mode == 'transition':
                    self.fade_in_alpha += self.alpha_step
                self.record_number += self.current_batch_size
                if self.level >= 4 and (self.record_number >= self.compute_swd_every or self.record_number == 0):
                    self.record_number = 0
                    self.swd.append(self.sample_swd())
                if self.passed_real_images_num >= self.switch_mode_number:
                    if self.mode == 'stabilize':
                        self.plot_stat_curve()
                        torch.save(self.G.state_dict(), f'state_dict/G_{2 ** self.level}.pkl')
                    self.update_state()
            self.train_G()
            if self.is_finished:
                break


if __name__ == '__main__':
    g = Generator(**config.__dict__)
    d = Discriminator(**config.__dict__)
    dataset = ImageDataset(data_root=config.data_root,
                           transform=T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])]),
                           max_resolution=config.resolution)
    pggan = PGGAN(g, d, dataset, **config.__dict__)
    pggan.train()
