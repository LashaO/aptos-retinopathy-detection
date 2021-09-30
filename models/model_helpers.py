import torch
import torch.nn as nn

import sys
sys.path.append('/home/l3404/Desktop/aptos2019-blindness-detection')
from utils.find_lr import LRFinder



############ FASTAI HEAD ##################

def listify(p=None, q=None):
    "Make `p` listy and the same length as `q`."
#     if p is None: p=[]
#     elif isinstance(p, str):          p = [p]
#     elif not isinstance(p, Iterable): p = [p]
    p = [p]

    # Rank 0 tensors in PyTorch are Iterable but don't have a length.
#     else:
#         try: a = len(p)
#         except: p = [p]
    n = q if type(q)==int else len(p) if q is None else len(q)
    if len(p)==1: p = p * n
    assert len(p)==n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(nn.Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"

    def __init__(self, full=False):
        super().__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)


def bn_drop_lin(n_in, n_out, bn=True, p=0.5, actn=None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers


def create_fastai_head(nf, nc, lin_ftrs=None, ps=0.5,
                concat_pool=True, bn_final=False):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes."
    lin_ftrs = [nf, 512, nc] if lin_ftrs is None else [nf] + lin_ftrs + [nc]
    ps = listify(ps)
    if len(ps) == 1: ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    layers = [pool, Flatten()]
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, True, p, actn)
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)

##################### END FASTAI HEAD #####################


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def set_grad_resnet(model, n=7, keep_bn=True, value=False):
    for name, param in model.named_parameters():
        n_layer = int(name.split('.')[1])
        if n_layer <= n and 'head' not in name:
            if keep_bn and 'bn' not in name:
                param.requires_grad = value


def print_grads(model, show_size=False):
  for name, param in model.named_parameters():
    if show_size:
      print(name, param.requires_grad, ' '*10, param.data.shape)
    else:
      print(name, param.requires_grad)


def find_lr(model, train_loader, optimizer, criterion, out_dir=None, device='cuda'):
    lr_finder = LRFinder(model, optimizer, criterion, device='cuda')
    lr_finder.range_test(train_loader, end_lr=0.1, num_iter=100, step_mode='exp')
    lr_finder.plot(log_lr=True, out_dir=out_dir)