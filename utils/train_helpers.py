#
# USAGE: python train.py --config=configs/resnet34.0.basic.yml
# tensorbordX: tensorboard --logdir='/home/l3404/Desktop/aptos2019-blindness-detection/results/resnet34.0.basic'
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import tqdm

import numpy as np
import torch
import torch.nn.functional as F

import utils
import utils.checkpoint
from utils.metrics import kappa_loss
from utils.utils import threshold_logits, seed_everything, inference


def evaluate_single_epoch(model, dataloader, criterion,
                          epoch, writer, postfix_dict, batch_size=32):
    model.eval()

    with torch.no_grad():
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        logit_list = []
        label_list = []
        loss_list = []
        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, data in tbar:
            images = data['image'].cuda()
            labels = data['label'].cuda()
            logits, aux_logits = inference(model, images)

            loss = criterion(logits, labels.float())
            if aux_logits is not None:
                aux_loss = criterion(aux_logits, labels.float())
                loss = loss + 0.4 * aux_loss
            loss_list.append(loss.item())

            logit_list.extend(logits.cpu().numpy())
            label_list.extend(labels.cpu().numpy())

            f_epoch = epoch + i / total_step
            desc = '{:5s}'.format('val')
            #desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
            tbar.set_description(desc)
            #tbar.set_postfix(**postfix_dict)

        log_dict = {}
        labels = np.array(label_list)
        logits = np.array(logit_list)

        predictions = threshold_logits(logits)
        accuracy = np.sum((predictions == labels).astype(float)) / float(predictions.size)

        log_dict['acc'] = accuracy
        log_dict['kappa'] = kappa_loss(predictions, labels)
        log_dict['loss'] = sum(loss_list) / len(loss_list)

        for key, value in log_dict.items():
            if writer is not None:
                writer.add_scalar('val/{}'.format(key), value, epoch)
            postfix_dict['val/{}'.format(key)] = value

        if writer is not None:
            for diagnosis in range(6):
                mask = (labels.astype(int) == diagnosis)

                kappa = kappa_loss(labels[mask], predictions[mask])
                writer.add_scalar('val/kappa_d={}'.format(diagnosis), kappa, epoch)

        print('val/epoch {} - loss: {}, kappa: {}, acc: {}'.format(epoch,
                                                                                 log_dict['loss'],
                                                                                 log_dict['kappa'],
                                                                                 log_dict['acc']))
        print('-' * 80)

        return log_dict['kappa']


def train_single_epoch(model, dataloader, criterion, optimizer,
                       epoch, writer, postfix_dict, batch_size=32, num_grad_acc=1):
    model.train()

    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    log_dict = {}
    log_avg_dict = {'loss': 0, 'acc': 0, 'kappa': 0}

    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    for i, data in tbar:
        images = data['image'].cuda()
        labels = data['label'].cuda()

        logits, aux_logits = inference(model, images)
        loss = criterion(logits, labels.float())
        if aux_logits is not None:
            aux_loss = criterion(aux_logits, labels.float())
            loss = loss + 0.4 * aux_loss
        log_dict['loss'] = loss.item()
        log_avg_dict['loss'] += loss.item()

        logits = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        predictions = threshold_logits(logits)
        accuracy = (predictions == labels).sum() / len(predictions)

        log_dict['acc'] = accuracy.item()
        log_dict['kappa'] = kappa_loss(predictions, labels)
        log_avg_dict['acc'] += accuracy.item()
        log_avg_dict['kappa'] += kappa_loss(predictions, labels)

        loss.backward()

        if (i+1) % num_grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()

        f_epoch = epoch + (i + 1) / total_step

        log_dict['lr'] = optimizer.param_groups[0]['lr']
        for key, value in log_dict.items():
            postfix_dict['train/{}'.format(key)] = value
            if writer is not None:
                writer.add_scalar('train/{}'.format(key), value, f_epoch * 1000)

        desc = '{:5s}'.format('train')
        # desc += ', {:04d}/{:04d}, {:.2f} epoch'.format(i + 1, total_step, f_epoch)
        tbar.set_description(desc)
        # tbar.set_postfix(**postfix_dict)

    log_avg_dict['loss'] /= total_step
    log_avg_dict['kappa'] /= total_step
    log_avg_dict['acc'] /= total_step

    print('train/epoch {} - loss: {}, kappa: {}, acc: {}'.format(epoch,
                                                                log_avg_dict['loss'],
                                                                log_avg_dict['kappa'],
                                                                log_avg_dict['acc']))
    for key, value in log_avg_dict.items():
        if writer is not None:
            writer.add_scalar('train_epoch/{}'.format(key), value, epoch)


def train(model, dataloaders, criterion, optimizer, scheduler, writer, start_epoch, num_epochs,
          batch_size=32, num_grad_acc=1, checkpoint_dir=None, best_checkpoint_only=False):

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    postfix_dict = {'train/loss': 0.0,
                    'train/kappa': 0.0,
                    'train/acc': 0.0,

                    'val/loss': 0.0,
                    'val/kappa': 0.0,
                    'val/acc': 0.0}

    kappa_list = []
    best_kappa = 0.0
    best_kappa_mavg = 0.0
    best_epoch = 0
    for epoch in range(start_epoch, num_epochs):
        # train phase
        train_single_epoch(model, dataloaders['train'],
                           criterion, optimizer, epoch, writer, postfix_dict,
                           batch_size=batch_size, num_grad_acc=num_grad_acc)

        # val phase
        kappa = evaluate_single_epoch(model, dataloaders['val'],
                                      criterion, epoch, writer, postfix_dict,
                                      batch_size=batch_size)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(kappa)
        else:
            scheduler.step()

        if (checkpoint_dir is not None) and (not best_checkpoint_only):
            utils.checkpoint.save_checkpoint(checkpoint_dir, model, optimizer, epoch, 0)

        kappa_list.append(kappa)
        kappa_tail = kappa_list[-10:]
        kappa_mavg = sum(kappa_tail) / len(kappa_tail)

        if kappa > best_kappa:
            best_kappa = kappa
            utils.checkpoint.save_checkpoint(checkpoint_dir, model, optimizer, epoch, 0,
                                             name='best_checkpoint')
        if kappa_mavg > best_kappa_mavg:
            best_kappa_mavg = kappa_mavg
    return {'best_kappa': best_kappa, 'best_kappa_mavg': best_kappa_mavg}


def predict(model, dataloader, df, batch_size=32, label='diagnosis', return_preds=False, output_filename=None):

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.eval()

    key_list = []
    label_list = []
    logit_list = []

    with torch.no_grad():
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, data in tbar:
            images = data['image'].cuda()
            logits, aux_logits = inference(model, images)

            logit_list.extend(logits.cpu().numpy())

        logits = np.array(logit_list)
        predictions = threshold_logits(logits)

        if return_preds:
            return predictions
        else:
            df[label] = predictions.astype(int)
            print('saved as {} in results/'.format(output_filename))
            df.to_csv(os.path.join(output_filename), index=False)




