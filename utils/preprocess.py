
# Resize and Crop
# Usage:
# python utils/preprocess.py --dir=data/train-original --new_size=256
import sys
sys.path.append('/home/l3404/Desktop/aptosnb')

import numpy as np
import glob
import os
from tqdm import tqdm
import cv2
from PIL import Image as PILImage ####
import argparse
from torchvision import transforms
from transforms.transform_factory import get_uncorr_transform



def crop_image_from_gray(image, tolerance=7):
    #image = np.array(image)

    if image.ndim == 2:
        mask = image > tolerance
        return image[np.ix_(mask.any(1), mask.any(0))]
    elif image.ndim == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = gray_image > tolerance

        check_shape = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return Image.fromarray(image)  # return original image
        else:
            image1 = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            image2 = image[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            image3 = image[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(image1.shape,image2.shape,image3.shape)
            image = np.stack([image1, image2, image3], axis=-1)
        #         print(image.shape)
        return image


# def resize2square(img, size, interpolation=cv2.INTER_AREA):
#     h, w = img.shape[:2]
#     c = None if len(img.shape) < 3 else img.shape[2]
#     if h == w:
#         return cv2.resize(img, (size, size), interpolation)
#     if h > w:
#         dif = h
#     else:
#         dif = w
#     x_pos = int((dif - w)/2.)
#     y_pos = int((dif - h)/2.)
#     if c is None:
#         mask = np.zeros((dif, dif), dtype=img.dtype)
#         mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
#     else:
#         mask = np.zeros((dif, dif, c), dtype=img.dtype)
#         mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
#     return cv2.resize(mask, (size, size), interpolation)


def resize2square(image, size):
    image = transforms.ToPILImage()(image)
    image = transforms.Resize(size)(image)
    image = transforms.CenterCrop((size, size))(image)
    #image = transforms.ToPILImage()(image)
    return image


def crop_and_resize(filepath, size):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = resize2square(image, size)
    return image


def parse_args():
    parser = argparse.ArgumentParser(description='APTOS')
    parser.add_argument('--dir', dest='image_dir',
                        help='configuration filename',
                        default=None, type=str)
    parser.add_argument('--new_size', dest='new_size',
                        help='configuration filename',
                        default=None, type=int)
    parser.add_argument('--dest', dest='dest_dir',
                        help='configuration filename',
                        default='data', type=str)
    return parser.parse_args()




def main():
    args = parse_args()

    image_dir = args.image_dir
    new_size = args.new_size
    dest_dir = args.dest_dir

    new_dir = image_dir.split('/')[-1] + '-uncorr-{}'.format(new_size)
    dest_dir = os.path.join(dest_dir, new_dir)
    print('Writing files to {}'.format(dest_dir))
    os.mkdir(dest_dir)

    filenames = os.listdir(image_dir)

    uncorr_transform = get_uncorr_transform(new_size)

    for i, filename in tqdm(enumerate(filenames)):

        filepath = os.path.join(image_dir, filename)
        image = PILImage.open(filepath)
        image = transforms.ToTensor()(image)
        try:
            image = uncorr_transform(image)
        except:
            print('could not transform image {}'.format(filename))

        image = transforms.ToPILImage()(image)

        image.save(os.path.join(dest_dir, filename))

    print('Done preprocessing the images!')


# def main():
#     args = parse_args()
#
#     image_dir = args.image_dir
#     new_size = args.new_size
#     dest_dir = args.dest_dir
#
#     new_dir = image_dir.split('/')[-1] + '-{}'.format(new_size)
#     dest_dir = os.path.join(dest_dir, new_dir)
#     print('Writing files to {}'.format(dest_dir))
#     os.mkdir(dest_dir)
#
#     filenames = os.listdir(image_dir)
#
#     for i, filename in tqdm(enumerate(filenames)):
#
#         filepath = os.path.join(image_dir, filename)
#
#         image = crop_and_resize(filepath, new_size)
#
#         image.save(os.path.join(dest_dir, filename))
#
#     print('Done preprocessing the images!')

if __name__ == '__main__':
    main()
