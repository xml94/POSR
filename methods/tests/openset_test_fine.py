import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import save_dir, root_model_path, root_criterion_path
from data.open_set_datasets import get_datasets, get_class_splits
from models.model_utils import get_model
from test.utils import EvaluateOpenSet, ModelTemplate
from utils.utils import str2bool
from utils.utils import strip_state_dict
from utils.utils import calculate_flops, calculate_FPS


class EnsembleModelEntropy(ModelTemplate):

    def __init__(self, all_models, mode='entropy', num_classes=4, use_softmax=False):

        super(ModelTemplate, self).__init__()

        self.all_models = all_models
        self.max_ent = torch.log(torch.Tensor([num_classes])).item()
        self.mode = mode
        self.use_softmax = use_softmax

    def entropy(self, preds):

        logp = torch.log(preds + 1e-5)
        entropy = torch.sum(-preds * logp, dim=-1)

        return entropy

    def forward(self, imgs):

        all_closed_set_preds = []

        for m in self.all_models:

            closed_set_preds = m(imgs)

            if self.use_softmax:
                closed_set_preds = torch.nn.Softmax(dim=-1)(closed_set_preds)

            all_closed_set_preds.append(closed_set_preds)

        closed_set_preds = torch.stack(all_closed_set_preds).mean(dim=0)

        if self.mode == 'entropy':
            open_set_preds = self.entropy(closed_set_preds)
        elif self.mode == 'max_softmax':
            open_set_preds = -closed_set_preds.max(dim=-1)[0]

        else:
            raise NotImplementedError

        return closed_set_preds, open_set_preds


def load_models(path, args):
    model = get_model(args, evaluate=True)

    if args.loss == 'ARPLoss':

        state_dict_list = [torch.load(p) for p in path]
        model.load_state_dict(state_dict_list)

    else:

        state_dict = strip_state_dict(torch.load(path[0]))
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def draw_curve(acc_all, save_path):
    thred = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99, 1.00]
    plt.xlabel("True positive rate")
    plt.ylabel("Open_set classification acc")
    # acc_avg = list(np.mean(acc_all, axis=0))
    acc_all_list = list(acc_all)
    # acc_all_list.append(acc_avg)
    x_ticks = np.arange(0.50, 1.00, 0.05)
    y_ticks = np.arange(0.00, 1.00, 0.05)
    name = ['Blueworms', 'Powdery_mildew', 'Chlorosis']
    for i in range(len(acc_all_list)):
        if i != len(acc_all_list)-1:
            # label_name = 'id %s' % i
            label_name = name[i]
            linewd = 1
        else:
            label_name = 'all'
            linewd = 2.5
        plt.plot(thred, acc_all_list[i], label=label_name, linewidth=linewd, marker='.')
    plt.legend()
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    save_path_ = save_path.replace('.svg', f"_acc_at_thresh_curve.svg")
    plt.savefig(save_path_)
    plt.clf()


def cal_mean(val, name):
    val = np.array(val)
    means = np.mean(val, axis=0)
    stds = np.std(val, axis=0)
    print(f"{name} mean is {means}, std is {stds}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='cls',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General
    parser.add_argument('--gpus', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3)')
    parser.add_argument('--pretrain', type=str, default='',
                        help='Which pretraining to use if --model=timm_resnet50_pretrained.'
                             'Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')
    parser.add_argument('--device', default='None', type=str, help='Which GPU to use')
    parser.add_argument('--osr_mode', default='max_softmax', type=str, help='{entropy, max_softmax}')
    parser.add_argument('--seed', default=0, type=int)

    # Model
    parser.add_argument('--model', type=str, default='classifier32')
    parser.add_argument('--loss', type=str, default='ARPLoss')
    parser.add_argument('--feat_dim', default=128, type=int)
    parser.add_argument('--max_epoch', default=109, type=int)
    parser.add_argument('--cs', default=False, type=str2bool)
    parser.add_argument('--admloss', default=False, type=str2bool)

    # vit
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Data params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--image_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dataset', type=str, default='ivadl_tomato')
    parser.add_argument('--transform', type=str, default='rand-augment')

    # Eval args
    parser.add_argument('--use_balanced_eval', default=False, type=str2bool)
    parser.add_argument('--use_softmax', default=False, type=str2bool)

    # Train params
    args = parser.parse_args()
    args.save_dir = save_dir
    device = torch.device('cuda:0')

    exp_ids = [
        '(05.07.2023_23.499)',
        '(05.07.2023_56.223)',
        '(05.07.2023_14.025)',
        '(05.07.2023_27.656)',
        '(05.07.2023_43.758)'
    ]

    # exp_ids = [
    #     '(03.07.2023_50.123)'
    # ]

    if args.cs:
        dataset_cs = args.dataset + 'cs'
    else:
        dataset_cs = args.dataset

    # Define paths
    all_paths_combined = [[x.format(i, args.dataset, dataset_cs, args.max_epoch, args.loss)
                           for x in (root_model_path, root_criterion_path)] for i in exp_ids]

    all_preds = []
    all_preds_acc_at_t = []
    fps_all, tps_all, fns_all, tns_all = [], [], [], []

    for split_idx in range(5):

        # ------------------------
        # DATASETS
        # ------------------------

        args.train_classes, args.open_set_classes = get_class_splits(args.dataset, split_idx=split_idx,
                                                                     cifar_plus_n=50)

        datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                image_size=args.image_size, balance_open_set_eval=args.use_balanced_eval,
                                split_train_val=False, open_set_classes=args.open_set_classes)

        # ------------------------
        # DATALOADERS
        # ------------------------
        dataloaders = {}
        for k, v, in datasets.items():
            shuffle = True if k == 'train' else False
            dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                        shuffle=shuffle, sampler=None, num_workers=args.num_workers)

        # ------------------------
        # MODEL
        # ------------------------
        print('Loading models...')

        all_models = [load_models(path=all_paths_combined[split_idx], args=args)]
        # calculate_flops(all_models[0])
        # calculate_FPS(all_models[0])

        model = EnsembleModelEntropy(all_models=all_models, mode=args.osr_mode,
                                     num_classes=len(args.train_classes), use_softmax=args.use_softmax)
        model.eval()
        model = model.to(device)

        # ------------------------
        # EVALUATE
        # ------------------------
        evaluate = EvaluateOpenSet(model=model, known_data_loader=dataloaders['test_known'],
                                   unknown_data_loader=dataloaders['test_unknown'], device=device,
                                   save_dir=args.save_dir, split_idx=split_idx,
                                   known_class=args.train_classes, unknown_class=args.open_set_classes,
                                   dataset=args.dataset)

        # Make predictions on test sets
        evaluate.predict()
        preds = evaluate.evaluate(evaluate)
        all_preds.append(preds[:len(preds) - 5])
        all_preds_acc_at_t.append(preds[-1])
        fps_all.append(preds[-2])
        tps_all.append(preds[-3])
        fns_all.append(preds[-4])
        tns_all.append(preds[-5])

    all_preds = np.array(all_preds)
    all_preds_acc_at_t = np.array(all_preds_acc_at_t)

    means = np.mean(all_preds, axis=0)
    stds = np.std(all_preds, axis=0)
    print(
        f'Mean ACC: {means[0]:.4f} pm {stds[0]:.4f} | Mean ACC_95: {means[1]:.4f} pm {stds[1]:.4f} | '
        f'AUROC: {means[2]:.4f} pm {stds[2]:.4f} |'
        f' OSCR: {means[4]:.4f} pm {stds[4]:.4f} | sil: {means[5]:.4f} pm {stds[5]:.4f} |'
        f' cal: {means[6]:.4f} pm {stds[6]:.4f} | dav: {means[7]:.4f} pm {stds[7]:.4f} |'

    )
    # print(
    #     f'Mean ACC: {means[0]:.4f} pm {stds[0]:.4f} | Mean ACC_95: {means[1]:.4f} pm {stds[1]:.4f} | '
    #     f'AUROC: {means[2]:.4f} pm {stds[2]:.4f} |'
    #     f' OSCR: {means[4]:.4f} pm {stds[4]:.4f} |'
    #
    # )
    filename_path = os.path.join(args.save_dir, "figure.svg")
    # draw_curve(all_preds_acc_at_t, filename_path)
    cal_mean(fps_all, 'fps')
    cal_mean(tps_all, 'tps')
    cal_mean(fns_all, 'fns')
    cal_mean(tns_all, 'tns')
