import torch
import math
from torchvision import transforms as T
from data.dataset import CelebA
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os
import time


def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def denorm(img: torch.Tensor):
    return (img / 2 + 0.5).clamp_(min=0, max=1)


def visualize(imgs):
    imgs = denorm(imgs)
    imshow(torchvision.utils.make_grid(imgs, nrow=int(math.sqrt(len(imgs)))))


from trained_models.model2.PGGAN import Generator
def load_model(level):
    G = Generator()
    print(G.load_state_dict(torch.load(f'trained_models/model2/state_dict/G_{2 ** level}.pkl')))
    G.cuda().eval()
    return G


def test(G, level, batch_size=16):
    assert level in range(2, G.R + 1)
    batch_size = batch_size
    x = torch.randn(size=(batch_size, G.latent_dim)).cuda()
    fake_imgs = G(x, level=level).cpu().detach()
    # real_imgs = dataset(batch_size, level=level)
    # visualize(real_imgs)
    visualize(fake_imgs)
    torch.cuda.empty_cache()


def save_samples(G, level, batch_size):
    x = torch.randn(size=(batch_size, G.latent_dim)).cuda()
    if not os.path.isdir(f'samples/{level}'):
        os.mkdir(f'samples/{level}')
    fake_imgs = denorm(G(x, level=level).cpu().detach())
    for img in fake_imgs:
        img = T.ToPILImage()(img.squeeze(0))
        img.save(f'samples/{level}/{time.time()}.jpg')


dataset = CelebA(transform=T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])]))


