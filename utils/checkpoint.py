from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import torch


def get_last_checkpoint(checkpoint_dir):
  checkpoints = [checkpoint
                 for checkpoint in os.listdir(checkpoint_dir)
                 if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth')]
  if checkpoints:
    return os.path.join(checkpoint_dir, list(sorted(checkpoints))[-1])
  return None


def get_initial_checkpoint(checkpoint_dir):
  return get_last_checkpoint(checkpoint_dir)


def get_checkpoint(checkpoint_dir, name):
  return os.path.join(checkpoint_dir, name)


def copy_last_n_checkpoints(checkpoint_dir, n, name):
  checkpoints = [checkpoint
                 for checkpoint in os.listdir(checkpoint_dir)
                 if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth')]
  checkpoints = sorted(checkpoints)
  for i, checkpoint in enumerate(checkpoints[-n:]):
    shutil.copyfile(os.path.join(checkpoint_dir, checkpoint),
                    os.path.join(checkpoint_dir, name.format(i)))


def load_checkpoint(model, optimizer, checkpoint):
  # return -1, -1 if no checkpoint
  if checkpoint is None:
    return -1, -1
  # print('load checkpoint from', checkpoint)
  checkpoint = torch.load(checkpoint)

  checkpoint_dict = {}
  for k, v in checkpoint['state_dict'].items():
    if 'num_batches_tracked' in k:
      continue
    if k.startswith('module.'):
      if True:
        checkpoint_dict[k[7:]] = v
      else:
        checkpoint_dict['feature_extractor.' + k[7:]] = v
    else:
      if True:
        checkpoint_dict[k] = v
      else:
        checkpoint_dict['feature_extractor.' + k] = v

  model.load_state_dict(checkpoint_dict) #, strict=False)

  if optimizer is not None:
    optimizer.load_state_dict(checkpoint['optimizer_dict'])

  step = checkpoint['step'] if 'step' in checkpoint else -1
  last_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else -1

  return last_epoch, step


def load_last_checkpoint(model, optimizer, checkpoint_dir):
    # load last checkpoint, return None if no checkpoint
    checkpoint = get_last_checkpoint(checkpoint_dir)
    # returns -1, -1 if checkpoint is None
    last_epoch, step = load_checkpoint(model, optimizer, checkpoint)
    print('from checkpoint: {} last epoch:{}'.format(checkpoint, last_epoch))
    return last_epoch, step


def save_checkpoint(checkpoint_dir, model, optimizer, epoch, step, weights_dict=None, name=None):

  if name:
    checkpoint_path = os.path.join(checkpoint_dir, '{}.pth'.format(name))
  else:
    checkpoint_path = os.path.join(checkpoint_dir, 'epoch_{:04d}.pth'.format(epoch))

  if weights_dict is None:
    weights_dict = {
      'state_dict': model.state_dict(),
      'optimizer_dict' : optimizer.state_dict(),
      'epoch' : epoch,
      'step' : step,
    }
  torch.save(weights_dict, checkpoint_path)
