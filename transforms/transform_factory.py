from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize
from albumentations.pytorch.transforms import ToTensor
import itertools

import numpy as np

import random
import cv2
import torch
from torchvision import transforms


#from policy_transform import policy_transform
#from tta_transform import tta_transform
#from .default_transform import default_transform


def get_transform(config, split, params=None):
  f = globals().get(config.transform.name)

  if params is not None:
    return f(split, **params)
  else:
    return f(split)


# def basic_transform(split,
#                      **kwargs):
#   if split == 'train':
#     transform = Compose([
#       Resize(224, 224),
#       ToTensor(),
#       Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#
#   else:
#     transform = Compose([
#       Resize(224, 224),
#       ToTensor(),
#       Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#
#   return transform

def basic_transform(split,
                     **kwargs):
  if split == 'train':
    transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    ])

  else:
    transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

  return transform

#########################################################################
####https://www.kaggle.com/fhopfmueller/removing-unwanted-correlations-in-training-public
#########################################################################

def crop_out_black(img):
  height, width = img.shape[1:]
  black = img[:, :int(height / 20), :int(width / 20)].mean(dim=(1, 2))
  rowmeans = img.mean(dim=1)
  linemeans = img.mean(dim=2)
  nonblack_rows = ((rowmeans - black[:, None]).max(dim=0)[0] > .02).nonzero()
  nonblack_lines = ((linemeans - black[:, None]).max(dim=0)[0] > .02).nonzero()
  try:
    left, right = nonblack_rows[0].item(), nonblack_rows[-1].item()
    upper, lower = nonblack_lines[0].item(), nonblack_lines[-1].item()
    img = img[:, upper:lower, left:right]
  except:
    print('crop out black didnt work')
  return img


def shapify_torch(img_data):
  height, width = img_data.shape[1:]
  black = img_data[:, :int(height / 20), :int(width / 20)].mean(
    dim=(1, 2))  # get average r, g, b of top-left corner as an estimate for black value
  mask = ((img_data - black[:, None, None]).max(dim=0)[0] > .02).to(
    img_data.dtype)  # note torch's max with argument dim returns (max, argmax)
  return torch.stack((mask, mask, mask))


def shapify_pil(img):
  img_data = transforms.ToTensor()(img)  # rgb, height, width. range [0, 1]
  img_data = shapify_torch(img_data)
  return transforms.ToPILImage()(img_data)


def center(img):
  # crop such that the center of mass of non-black pixels is roughly in the center
  _, height, width = img.shape
  shapified = shapify_torch(img)[0, ...]  # just take one of 3 channels
  nonzero = (shapified).nonzero().to(torch.float)
  center = nonzero.mean(dim=0).to(torch.int)
  if center[0] > height / 2:  # center too low, crop from top
    new_height = 2 * (height - center[0])
    img = img[:, -new_height:, :]
  else:  # center too high, crop from bottom
    new_height = 2 * center[0]
    img = img[:, :new_height, :]
  if center[1] > width / 2:  # center too far right, crop from left
    new_width = 2 * (width - center[1])
    img = img[:, :, -new_width:]
  else:  # center too far left, crop from right
    new_width = 2 * center[1]
    img = img[:, :, :new_width]
  return img


def tight_crop(img):  # assumes black cropped out and centered
  shapified = shapify_torch(img)[0, ...]
  if shapified.to(torch.float).mean() > .95:  # already tight crop
    # print('already tight crop, passing')
    return img
  width = img.shape[2]
  width_margin = int(.06 * width)
  img = img[:, :, width_margin:-width_margin]
  shapified = shapified[:, width_margin:-width_margin]
  num_white_per_line = shapified.sum(dim=1) / shapified.shape[1]
  white_above_threshold = (num_white_per_line > .9).nonzero()
  if white_above_threshold.shape[0] < 10:
    #print('could not crop')
    return(img)
  upper, lower = white_above_threshold[0], white_above_threshold[-1]
  img = img[:, upper:lower, :]
  return img


def remove_corners(img):  # blacken a triangle of 1/6 at each corner. assumes square input
  corner_size = img.shape[1] // 6
  mask = torch.ones((corner_size, corner_size)).triu()
  img[:, :corner_size, :corner_size] *= mask.flip(dims=(0,))[None, :, :]
  img[:, :corner_size, -corner_size:] *= mask.flip(dims=(0, 1))[None, :, :]
  img[:, -corner_size:, :corner_size] *= mask[None, :, :]
  img[:, -corner_size:, -corner_size:] *= mask.flip(dims=(1,))[None, :, :]
  return (img)


def gaussian_blur(img, radius=None, rel_size=None):
  if radius is None:
    radius = int(rel_size * img.shape[1])
  if radius % 2 == 0:
    radius = radius + 1
  img_numpy = img.permute(1, 2, 0).numpy()
  img_numpy = cv2.GaussianBlur(img_numpy, (radius, radius), 0)
  img = torch.Tensor(img_numpy).permute(2, 0, 1)
  return img


def subtract_gaussian_blur(img, rel_size=.2, color_scale=1):
  img_blurred = gaussian_blur(img, rel_size=rel_size)
  img = (4 * color_scale * (
            img - img_blurred)).sigmoid()  # sigmoid to squish to [0, 1]. Factor 4 because the slope of sigmoid at 0 is 4.
  return img


# def visualize_transform(transform, compare=False, train=True):
#   """
#   Visualizes a transformation, with one example per original size.
#   Parameters:
#   - transform: the transformation. Should take a pytorch tensor to a pytorch tensor.
#   - compare: if true, show the unmodified and transformed images next to each other
#   - train:
#   """
#   if train:
#     unique_dims = train_unique_dims
#     samples_per_dim = train_samples_per_dim
#     df = train_df
#     path = train_path
#   else:
#     unique_dims = test_unique_dims
#     samples_per_dim = test_samples_per_dim
#     df = test_df
#     path = test_path
#   if compare:
#     fig, axs = plt.subplots(8, 5, figsize=(20, 30))
#   else:
#     fig, axs = plt.subplots(4, 5, figsize=(20, 15))
#
#   for i, unique_dim in enumerate(unique_dims):
#     row = 2 * (i // 5) if compare else i // 5
#     img_idx = np.random.choice(samples_per_dim[unique_dim])
#     id_code = df[img_idx][0]
#     img = transforms.ToTensor()(Image.open(path + id_code + '.png'))
#     axs[row][i % 5].imshow(transforms.ToPILImage()(transform(img)))
#     if train:
#       axs[row][i % 5].set_title(f'#{i}. {unique_dim}. {train_histogram_per_dim[unique_dim]}. ' +
#                                 f'{100 * sum(train_histogram_per_dim[unique_dim]) / len(train_df):.2f}%', fontsize=8)
#     else:
#       axs[row][i % 5].set_title(f'#{i}. {unique_dim}. ' +
#                                 f'{100 * len(samples_per_dim[unique_dim]) / len(df):.2f}%', fontsize=8)
#     if compare:
#       axs[row + 1][i % 5].imshow(transforms.ToPILImage()(img))

def get_uncorr_transform(image_size):
  transform = transforms.Compose([
    # lambda img: torch.nn.functional.interpolate(img[None, ...], size=(image_size, image_size), mode='bilinear',
    #                                             align_corners=False)[0, ...],
    crop_out_black,
    center,
    tight_crop,
    lambda img: torch.nn.functional.interpolate(img[None, ...], size=(image_size, image_size), mode='bilinear',
                                                align_corners=False)[0, ...],
    lambda img: subtract_gaussian_blur(img, rel_size=.2, color_scale=2),
    remove_corners])

  return transform

def get_uncorr_transform_noben(image_size):
  transform = transforms.Compose([
    lambda img: torch.nn.functional.interpolate(img[None, ...], size=(image_size, image_size), mode='bilinear',
                                                align_corners=False)[0, ...],
    crop_out_black,
    center,
    tight_crop,
    #lambda img: subtract_gaussian_blur(img, rel_size=.2, color_scale=2),
    remove_corners])

  return transform


#########################################################################
#########################################################################
#########################################################################