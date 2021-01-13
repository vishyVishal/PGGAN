import random
import os
from PIL import Image
import numpy as np
import torch


class CelebA(object):
    def __init__(self, transform=None):
        self.database = 'data/celeba_faces'
        self.images = []
        self.reload_images()
        self._length = len(self.images)
        self.transform = transform

    def reload_images(self):
        self.images = os.listdir(f'{self.database}/4/data')
        random.shuffle(self.images)

    def __call__(self, batch_size, level):
        assert level in range(2, 9)
        resolution = 2 ** level
        dir_path = os.path.join(self.database, str(resolution), 'data')
        if len(self.images) < batch_size:
            self.reload_images()
        img_files, self.images = self.images[:batch_size], self.images[batch_size:]
        if self.transform is not None:
            imgs = torch.stack([self.transform(Image.open(os.path.join(dir_path, img))) for img in img_files])
        else:
            imgs = np.stack([np.array(Image.open(os.path.join(dir_path, img))) for img in img_files]).transpose((0, 3, 1, 2))
        return imgs

    def __len__(self):
        return self._length