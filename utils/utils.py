from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torchvision
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

import scipy
from sklearn.metrics import cohen_kappa_score
from functools import partial


import cv2
import numpy as np

def prepare_train_directories(config):
  out_dir = config.train.dir
  os.makedirs(os.path.join(out_dir, 'checkpoint'), exist_ok=True)


# def threshold_logits(X, coef=None):
#   if coef is None:
#     coef = torch.tensor([0.5, 1.5, 2.5, 3.5]) # initial coef
#
#   X_p = X.clone().detach().cpu()
#   for i, pred in enumerate(X_p):
#     if pred < coef[0]:
#       X_p[i] = 0
#     elif pred >= coef[0] and pred < coef[1]:
#       X_p[i] = 1
#     elif pred >= coef[1] and pred < coef[2]:
#       X_p[i] = 2
#     elif pred >= coef[2] and pred < coef[3]:
#       X_p[i] = 3
#     else:
#       X_p[i] = 4
#
#   return X_p.long()

def prepare_train_directories(out_dir):
  os.makedirs(os.path.join(out_dir, 'checkpoint'), exist_ok=True)


def threshold_logits(X, coef=None):
  if coef is None:
    coef = [0.5, 1.5, 2.5, 3.5] # initial coef

  X_p = np.copy(X)
  for i, pred in enumerate(X_p):
    if pred < coef[0]:
      X_p[i] = 0
    elif pred >= coef[0] and pred < coef[1]:
      X_p[i] = 1
    elif pred >= coef[1] and pred < coef[2]:
      X_p[i] = 2
    elif pred >= coef[2] and pred < coef[3]:
      X_p[i] = 3
    else:
      X_p[i] = 4

  return X_p.astype(int)

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = scipy.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        print(-loss_partial(self.coef_['x']))

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']

def seed_everything(seed=808):
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def inference(model, images):
  logits = model(images)
  if isinstance(logits, tuple):
    logits, aux_logits = logits
  else:
    aux_logits = None
  # probabilities = F.sigmoid(logits)
  return logits.squeeze(1), aux_logits  # , probabilities


def get_split_indices(data, target_column='label', random_state=0):
  skf = StratifiedKFold(n_splits=5, random_state=random_state)
  try:
    y = data[target_column].values
  except KeyError:
    raise KeyError('target_column is - {}'.format(target_column))
  X = np.arange(len(y))
  folds = []

  for fold_idx, (train_index, val_index) in enumerate(skf.split(X, y)):
    folds.append((train_index, val_index))

  return folds


def plot_batch(dataloader):
  batch = next(iter(dataloader))['image']
  grid_img = torchvision.utils.make_grid(batch, nrow=4, padding=16, pad_value=1)
  plt.figure(figsize=(16, 32))
  plt.imshow(grid_img.permute(1, 2, 0))



