from PIL import Image
import os
import tqdm


if __name__ == '__main__':
    total_images = os.listdir('flickr/256/data')
    for image_path in tqdm.tqdm(total_images):
        img = Image.open(os.path.join('flickr/256/data', image_path))
        for resolution in [4, 8, 16, 32, 64, 128]:
            save_path = os.path.join('flickr', str(resolution), 'data', image_path)
            img.resize(size=(resolution, resolution)).save(save_path)
        img.close()
