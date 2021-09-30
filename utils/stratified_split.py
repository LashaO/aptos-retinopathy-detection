from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold


def split_stratified(data_dir, train, external=False, target_column='diagnosis'):
    skf = StratifiedKFold(n_splits=5, random_state=0)
    y = train[target_column].values
    X = np.arange(len(y))
    folds = []

    if external:
        splitname = 'split.stratified.big'
        print('Using external data.')
    else:
        splitname = 'split.stratified.small'
        print('Not using external data.')

    for fold_idx, (train_index, val_index) in enumerate(skf.split(X, y)):
        folds.append((train_index, val_index))

        train_fold = train.loc[train_index]
        val_fold = train.loc[val_index]

        train_fold['split'] = 'train'
        val_fold['split'] = 'val'

        train = pd.concat([train_fold, val_fold])

        filename = os.path.join(data_dir, 'splits', (splitname + '.{}.csv'.format(fold_idx)))
        train.to_csv(filename, index=False)

    print('Done splitting folds...')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', dest='data_dir',
                        help='the directory of the data',
                        default='data', type=str)

    parser.add_argument('--use_external', dest='use_external',
                        help='1: with external, 0: without external',
                        default=0, type=int)

    return parser.parse_args()


def main():
    args = parse_args()

    num_fold = 5
    data_dir = args.data_dir
    use_external = args.use_external == 1

    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    train['source'] = 'train'
    if use_external:
        external = pd.read_csv(os.path.join(data_dir, 'external.csv'))
        external['source'] = 'external'
        train = pd.concat([train, external]).reset_index()

    split_stratified(data_dir, train, external=use_external)


if __name__ == '__main__':
  main()

