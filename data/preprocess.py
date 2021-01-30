from PIL import Image
import os
import tqdm
import argparse


parser = argparse.ArgumentParser('Choose Data Root')
parser.add_argument('--data_root', '-dr', type=str, default='celeba_faces')
args, _ = parser.parse_known_args()
if __name__ == '__main__':
    total_images = os.listdir(os.path.join(args.data_root, '256/data'))
    for image_path in tqdm.tqdm(total_images):
        img = Image.open(os.path.join(args.data_root, '256/data', image_path))
        for resolution in [4, 8, 16, 32, 64, 128]:
            if not os.path.isdir(os.path.join(args.data_root, str(resolution), 'data')):
                os.mkdir(os.path.join(args.data_root, str(resolution), 'data'))
            save_path = os.path.join(args.data_root, str(resolution), 'data', image_path)
            img.resize(size=(resolution, resolution)).save(save_path)
        img.close()
