from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class TestDataset(Dataset):
    def __init__(self,
                 df,
                 images_dir,
                 transform=None,
                 **_):
        self.df = df
        self.images_dir = images_dir
        self.transform = transform

        self.df_filenames = self.load_filenames()
        self.examples = self.load_examples()
        self.size = len(self.examples)

    def load_filenames(self):
        df_filenames = self.df.copy()

        def to_filepath(name):
            return os.path.join(self.images_dir, name + '.png')

        df_filenames['filepath'] = df_filenames['id_code'].transform(to_filepath)
        return df_filenames

    def load_examples(self):
        return [(row['id_code'], row['filepath'], torch.tensor(int(row['diagnosis'])))
                for _, row in self.df_filenames.iterrows()]

    def __getitem__(self, index):
        example = self.examples[index]

        filepath = example[1]
        image = Image.open(filepath)

        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

        image = self.transform(image)

        return {'image': image,
                'key': example[0]}

    def __len__(self):
        return self.size


def test():
    pass


if __name__ == '__main__':
    test()

