from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tqdm
import numpy as np
import pandas as pd
import scipy.misc as misc
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset

from torchvision import transforms


class BigDataset(Dataset):
    def __init__(self,
                 df,
                 images_dir,
                 images_external_dir,
                 transform=None,
                 **_):
        self.df = df
        self.images_dir = images_dir
        self.images_external_dir = images_external_dir
        self.transform = transform

        self.df_labels = self.load_labels()
        self.examples = self.load_examples()
        self.size = len(self.examples)

    def load_labels(self):
        df_labels = self.df.copy()

        def to_filepath(name):
            if len(name) == len('0104b032c141'):
                return os.path.join(self.images_dir, name + '.png')
            else:
                return os.path.join(self.images_external_dir, name + '.jpeg')

        df_labels['filepath'] = df_labels['name'].transform(to_filepath)
        return df_labels

    def load_examples(self):
        return [(row['name'], row['filepath'], torch.tensor(int(row['label']))) \
                for _, row in self.df_labels.iterrows()]

    def __getitem__(self, index):
        example = self.examples[index]

        filepath = example[1]
        #print('FILEPATH IS', filepath)
        label = example[2]
        image = Image.open(filepath)

        # Onehot
        # label = [0 for _ in range(28)]
        # for l in example[2]:
        #    label[l] = 1
        # label = np.array(label)

        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

        image = self.transform(image)

        return {'image': image,
                'label': label,
                'key': example[0]}

    def __len__(self):
        return self.size


def test():
    pass


if __name__ == '__main__':
    test()
