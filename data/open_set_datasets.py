
from data.cottonweed import get_cotton_weed_datasets
from data.ivadl_tomato import get_ivadl_tomato_datasets
from data.ivadl_rose import get_ivadl_rose_datasets

from data.paddy_rice import get_paddy_rice_datasets
from data.open_set_splits.osr_splits import osr_splits
from data.augmentations import get_transform

import os
import sys
import pickle


"""
For each dataset, define function which returns:
    training set
    validation set
    open_set_known_images
    open_set_unknown_images
"""

get_dataset_funcs = {
    'cotton-weed': get_cotton_weed_datasets,
    'ivadl_tomato': get_ivadl_tomato_datasets,
    'ivadl_rose': get_ivadl_rose_datasets,
    'paddy_rice': get_paddy_rice_datasets
}


def get_datasets(name, transform='default', image_size=224, train_classes=(0, 1, 8, 9),
                 open_set_classes=range(10), balance_open_set_eval=False, split_train_val=True, seed=0, args=None):
    """
    :param name: Dataset name
    :param transform: Either tuple of train/test transforms or string of transform type
    :return:
    """

    print('Loading datasets...')

    if isinstance(transform, tuple):
        train_transform, test_transform = transform
    else:
        train_transform, test_transform = get_transform(transform_type=transform, image_size=image_size, args=args)

    if name in get_dataset_funcs.keys():
        datasets = get_dataset_funcs[name](train_transform, test_transform,
                                           train_classes=train_classes,
                                           open_set_classes=open_set_classes,
                                           balance_open_set_eval=balance_open_set_eval,
                                           split_train_val=split_train_val,
                                           seed=seed)
    else:
        raise NotImplementedError

    return datasets


def get_class_splits(dataset, split_idx=0, cifar_plus_n=10):

    if dataset == 'cotton-weed':
        # train_classes = osr_splits[dataset][split_idx]
        # open_set_classes = [x for x in range(15) if x not in train_classes]

        # for cotton-weed dataset, the unknown classes comes from osr_splites
        open_set_classes = osr_splits[dataset][split_idx]
        train_classes = [x for x in range(15) if x not in open_set_classes]

    elif dataset == 'ivadl_tomato':
        # train_classes = osr_splits[dataset][split_idx]
        # open_set_classes = [x for x in range(9) if x not in train_classes]

        # for tomato dataset, the unknown classes comes from osr_splites
        open_set_classes = osr_splits[dataset][split_idx]
        train_classes = [x for x in range(9) if x not in open_set_classes]

    elif dataset == 'ivadl_rose':
        train_classes = osr_splits[dataset][split_idx]
        open_set_classes = [x for x in range(6) if x not in train_classes]
        # open_set_classes = osr_splits[dataset][split_idx]
        # train_classes = [x for x in range(6) if x not in open_set_classes]
    elif dataset == 'paddy_rice':
        # train_classes = osr_splits[dataset][split_idx]
        # open_set_classes = [x for x in range(10) if x not in train_classes]
        open_set_classes = osr_splits[dataset][split_idx]
        train_classes = [x for x in range(10) if x not in open_set_classes]

    else:

        raise NotImplementedError

    return train_classes, open_set_classes


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
