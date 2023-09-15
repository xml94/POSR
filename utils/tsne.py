import torch
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def visualize_tsne_points(tx, ty, labels, save_path, epoch, flag, classes, dataset, unknown_in_one):
    colors_per_class_old = get_colors_per_class(dataset)
    file_name = os.path.join(save_path, str(epoch) + flag + "_tsne_figure.svg")
    colors_per_class = {}
    class_name = get_class_name(dataset)
    new_class_name = [class_name[x] for x in classes]
    for name in new_class_name:
        colors_per_class.update(
            {
                name: colors_per_class_old[name]
            }
        )
    if unknown_in_one:
        for name in new_class_name[-3:]:
            colors_per_class.update(
                {
                    name: [0, 0, 0]
                }
            )
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label, key in enumerate(colors_per_class):
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[key][::-1]], dtype=np.float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=key)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    # plt.show()
    plt.savefig(file_name)
    plt.clf()


def visualize_tsne(features, labels, save_dir, epoch, flag, classes, dataset, unknown_in_one=False):
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(features)
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels, save_dir, epoch, flag, classes, dataset, unknown_in_one)


def get_colors_per_class(dataset):
    colors_per_class_old = {}
    if dataset == 'ivadl_tomato':
        colors_per_class_old = {
            'Ulcer': [254, 202, 87],
            'Leaf_fungus': [255, 107, 107],
            'Septoria_spot': [10, 189, 227],
            'Chlorosis': [255, 159, 243],
            'Yellow_curl': [16, 172, 132],
            'Powdery_mildew': [128, 80, 128],
            'Healthy': [87, 101, 116],
            'Leaf_miner': [52, 31, 151],
            'Blueworms': [100, 100, 255],
        }
    elif dataset == 'ivadl_rose':
        colors_per_class_old = {
            'Rose_D04': [254, 202, 87],
            'Rose_H': [255, 107, 107],
            'Rose_P01': [10, 189, 227],
            'Rose_P02': [255, 159, 243],
            'Rose_R01': [16, 172, 132],
            'Rose_R02': [128, 80, 128],
        }
    elif dataset == 'apple2021':
        colors_per_class_old = {
            'complex': [254, 202, 87],
            'frog_eye_spot': [255, 107, 107],
            'healthy': [10, 189, 227],
            'powdery_mildew': [255, 159, 243],
            'rust': [16, 172, 132],
            'scab': [128, 80, 128],
        }
    elif dataset == 'cotton-weed':
        colors_per_class_old = {
            'Carpetweeds': [254, 202, 87],
            'Crabgrass': [255, 107, 107],
            'Eclipta': [10, 189, 227],
            'Goosegrass': [255, 159, 243],
            'Morningglory': [16, 172, 132],
            'Nutsedge': [128, 80, 128],
            'PalmerAmaranth': [87, 101, 116],
            'PricklySida': [52, 31, 151],
            'Purslane': [100, 100, 255],
            'Ragweed': [255, 0, 0],
            'Sicklepod': [255, 255, 0],
            'SpottedSpurge': [0, 128, 0],
            'SpurredAnoda': [0, 0, 255],
            'Swinecress': [255, 0, 255],
            'Waterhemp': [128, 128, 0],
        }
    elif dataset == 'paddy_rice':
        colors_per_class_old = {
            'BLB': [254, 202, 87],
            'BLS': [255, 107, 107],
            'BPB': [10, 189, 227],
            'BS': [255, 159, 243],
            'Blast': [16, 172, 132],
            'DH': [128, 80, 128],
            'DM': [87, 101, 116],
            'Healthy': [52, 31, 151],
            'Hispa': [100, 100, 255],
            'Tungro': [255, 0, 0],
        }

    return colors_per_class_old


def get_class_name(dataset):
    class_name = []
    if dataset == 'ivadl_tomato':
        class_name = ['Ulcer', 'Leaf_fungus', 'Septoria_spot',
                      'Chlorosis', 'Yellow_curl', 'Powdery_mildew',
                      'Healthy', 'Leaf_miner', 'Blueworms']
    elif dataset == 'ivadl_rose':
        class_name = ['Rose_D04', 'Rose_H', 'Rose_P01', 'Rose_P02', 'Rose_R01', 'Rose_R02']
    elif dataset == 'apple2021':
        class_name = ['complex', 'frog_eye_spot', 'healthy',
                      'powdery_mildew', 'rust', 'scab']
    elif dataset == 'cotton-weed':
        class_name = ['Carpetweeds', 'Crabgrass', 'Eclipta', 'Goosegrass', 'Morningglory', 'Nutsedge', 'PalmerAmaranth',
                      'PricklySida', 'Purslane', 'Ragweed', 'Sicklepod', 'SpottedSpurge', 'SpurredAnoda', 'Swinecress',
                      'Waterhemp']
    elif dataset == 'paddy_rice':
        class_name = ['BLB', 'BLS', 'BPB', 'BS', 'Blast', 'DH', 'DM', 'Healthy', 'Hispa', 'Tungro']

    return class_name


if __name__ == '__main__':
    main()
