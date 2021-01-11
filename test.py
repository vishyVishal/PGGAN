import torch
import math
from model.PGGAN import Generator, Discriminator
from torchvision import transforms as T
from data.dataset import CelebA
import numpy as np
import matplotlib.pyplot as plt
import torchvision


def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def denorm(img):
    return img / 2 + 0.5


def visualize(imgs):
    imgs = denorm(imgs)
    imshow(torchvision.utils.make_grid(imgs, nrow=int(math.sqrt(len(imgs)))))


def load_model(level):
    assert level in range(2, 9)
    G = Generator()
    print(G.load_state_dict(torch.load('state_dict/G_4.pkl')))
    G.cuda().eval()
    return G


def test(G, level, batch_size=16):
    assert level in range(2, 9)
    batch_size = batch_size
    x = torch.randn(size=(batch_size, 512)).cuda()
    fake_imgs = G(x, level=level).cpu().detach()
    real_imgs = dataset(batch_size, level=level)
    visualize(real_imgs)
    visualize(fake_imgs)
    torch.cuda.empty_cache()


dataset = CelebA(transform=T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])]))


