from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# pylint: disable=W0611
import types

from fastai import *
from fastai.vision import *

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
# import pretrainedmodels
from models import *
from .model_helpers import create_fastai_head, Identity



def get_resnet50(num_classes=1, pretrained=True, **_):

    if pretrained:
        print('Attempting to download pretrained model...')

    model = torchvision.models.resnet50(pretrained=pretrained)
    num_features = model.fc.in_features

    model.fc = nn.Linear(num_features, num_classes)

    return model


def get_resnet50_fastai(num_classes=1, pretrained=True):
    body = create_body(models.resnet50, pretrained=pretrained, cut=None)
    nf = callbacks.hooks.num_features_model(body) * 2
    head = create_head(nf, num_classes, lin_ftrs=None, ps=0.5, bn_final=False)

    model = nn.Sequential(OrderedDict([
        ('body', body),
        ('head', head)])).cuda()

    return model.cuda()


def get_model(config):
    print('model name:', config.model.name)
    f = globals().get('get_' + config.model.name)
    if config.model.params is None:
        return f()
    else:
        return f(**config.model.params)


if __name__ == '__main__':
    print('model_factory main()')
    pass
    #print('main')
    #model = get_resnet34()

# def get_attention_inceptionv3(num_classes=28, **kwargs):
#     return AttentionInceptionV3(num_classes=num_classes, **kwargs)
#
#
# def get_attention(num_classes=28, **kwargs):
#     return Attention(num_classes=num_classes, **kwargs)
#
#
# def get_resnet34(num_classes=28, **_):
#     model_name = 'resnet34'
#     model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
#     conv1 = model.conv1
#     model.conv1 = nn.Conv2d(in_channels=4,
#                             out_channels=conv1.out_channels,
#                             kernel_size=conv1.kernel_size,
#                             stride=conv1.stride,
#                             padding=conv1.padding,
#                             bias=conv1.bias)
#
#     # copy pretrained weights
#     model.conv1.weight.data[:,:3,:,:] = conv1.weight.data
#     model.conv1.weight.data[:,3:,:,:] = conv1.weight.data[:,:1,:,:]
#
#     model.avgpool = nn.AdaptiveAvgPool2d(1)
#     in_features = model.last_linear.in_features
#     model.last_linear = nn.Linear(in_features, num_classes)
#     return model
#
#
# def get_resnet18(num_classes=28, **_):
#     model_name = 'resnet18'
#     model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
#     conv1 = model.conv1
#     model.conv1 = nn.Conv2d(in_channels=4,
#                             out_channels=conv1.out_channels,
#                             kernel_size=conv1.kernel_size,
#                             stride=conv1.stride,
#                             padding=conv1.padding,
#                             bias=conv1.bias)
#
#     # copy pretrained weights
#     model.conv1.weight.data[:,:3,:,:] = conv1.weight.data
#     model.conv1.weight.data[:,3:,:,:] = conv1.weight.data[:,:1,:,:]
#
#     model.avgpool = nn.AdaptiveAvgPool2d(1)
#     in_features = model.last_linear.in_features
#     model.last_linear = nn.Linear(in_features, num_classes)
#     return model
#
#
# def get_senet(model_name='se_resnext50', num_classes=28, **_):
#     model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
#     conv1 = model.layer0.conv1
#     model.layer0.conv1 = nn.Conv2d(in_channels=4,
#                                    out_channels=conv1.out_channels,
#                                    kernel_size=conv1.kernel_size,
#                                    stride=conv1.stride,
#                                    padding=conv1.padding,
#                                    bias=conv1.bias)
#
#     # copy pretrained weights
#     model.layer0.conv1.weight.data[:,:3,:,:] = conv1.weight.data
#     model.layer0.conv1.weight.data[:,3:,:,:] = conv1.weight.data[:,:1,:,:]
#
#     model.avgpool = nn.AdaptiveAvgPool2d(1)
#     in_features = model.last_linear.in_features
#     model.last_linear = nn.Linear(in_features, num_classes)
#     return model


# def get_se_resnext50(num_classes=28, **kwargs):
#     return get_senet('se_resnext50_32x4d', num_classes=num_classes, **kwargs)