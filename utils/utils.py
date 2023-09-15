import os
import torch
import random
import numpy as np
import inspect
import cv2
import pandas as pd
import torch.nn as nn
import time

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from config import project_root_dir
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def strip_state_dict(state_dict, strip_key='module.'):
    """
    Strip 'module' from start of state_dict keys
    Useful if model has been trained as DataParallel model
    """

    for k in list(state_dict.keys()):
        if k.startswith(strip_key):
            state_dict[k[len(strip_key):]] = state_dict[k]
            del state_dict[k]

    return state_dict


def init_experiment(args, runner_name=None):
    args.cuda = torch.cuda.is_available()

    if args.device == 'None':
        args.device = torch.device("cuda:0" if args.cuda else "cpu")
    else:
        args.device = torch.device(args.device if args.cuda else "cpu")

    print(args.gpus)

    # Get filepath of calling script
    if runner_name is None:
        runner_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(".")[-2:]

    root_dir = os.path.join(args.exp_root, *runner_name)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Unique identifier for experiment
    now = '({:02d}.{:02d}.{}_'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
          datetime.now().strftime("%S.%f")[:-3] + ')'

    log_dir = os.path.join(root_dir, 'log', now)
    while os.path.exists(log_dir):
        now = '({:02d}.{:02d}.{}_'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
              datetime.now().strftime("%S.%f")[:-3] + ')'

        log_dir = os.path.join(root_dir, 'log', now)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.log_dir = log_dir

    # Instantiate directory to save models to
    model_root_dir = os.path.join(args.log_dir, 'checkpoints')
    os.mkdir(model_root_dir)

    args.model_dir = model_root_dir

    print(f'Experiment saved to: {args.log_dir}')

    args.writer = SummaryWriter(log_dir=args.log_dir)

    hparam_dict = {}

    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparam_dict[k] = v

    args.writer.add_hparams(hparam_dict=hparam_dict, metric_dict={})

    print(runner_name)
    print(args)

    return args


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_default_hyperparameters(args):
    """
    Adjusts args to match parameters used in paper: https://arxiv.org/abs/2110.06207
    """

    hyperparameter_path = os.path.join(project_root_dir, 'utils/paper_hyperparameters.csv')
    df = pd.read_csv(hyperparameter_path)

    df = df.loc[df['Loss'] == args.loss]
    hyperparams = df.loc[df['Dataset'] == args.dataset].values[0][2:]

    # -----------------
    # DATASET / LOSS specific hyperparams
    # -----------------
    args.image_size, args.lr, args.rand_aug_n, args.rand_aug_m, args.label_smoothing, args.batch_size = hyperparams

    if args.dataset in ('cub', 'aircraft', 'scars', 'imagenet'):

        args.model = 'timm_resnet50_pretrained'
        args.resnet50_pretrain = 'places_moco'
        args.feat_dim = 2048

    else:

        args.model = 'classifier32'
        args.feat_dim = 128

    # -----------------
    # Other hyperparameters
    # -----------------
    args.seed = 0
    args.max_epoch = 600
    args.transform = 'rand-augment'

    args.scheduler = 'cosine_warm_restarts_warmup'
    args.num_restarts = 2
    args.weight_decay = 1e-4

    return args


def calculate_flops(model):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        if isinstance(model, (nn.parallel.distributed.DistributedDataParallel, nn.DataParallel)):
            model = model.module
        # input = cv2.imread("./datasets/CottonWeedID15/raw/val/Carpetweeds"
        #                    "/eea0b961233245e996b0d3737ca00940__carpetweed__3.jpg")
        input = cv2.imread("./datasets/PaddyDoctor10407/raw/val/BLB"
                           "/110079.jpg")

        input = cv2.resize(input, (224, 224))
        input = torch.from_numpy(input).permute(2, 0, 1)
        input = input[None, :, :, :].float()
        input = input.to(device)

        flop = FlopCountAnalysis(model, input)
        print(flop_count_table(flop, max_depth=4))
        print(flop_count_str(flop))
        print(flop.total())


def calculate_FPS(model):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model = model.to(device)

    total_time = []
    input = cv2.imread("./datasets/IVADL_tomato/raw/val/Tomato_D01_ulcer"
                           "/0017104_SAM_2484.png")
    # input = cv2.imread("./datasets/IVADL_rose/raw/val/Rose_D04/0022600_SAM_3316.png")

    input = cv2.resize(input, (224, 224))
    input = torch.from_numpy(input).permute(2, 0, 1)
    input = input[None, :, :, :].float()
    input = input.to(device)
    iteration = 10000

    # Measure performance
    with torch.no_grad():
        # start_time = time.time()
        for i in range(iteration):
            # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # starter.record()
            start_time = time.time()
            _ = model(input)
            # ender.record()
            curr_time = time.time() - start_time
            # wait for GPU SYNC
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender) / 1000
            total_time.append(curr_time)
        # curr_time = time.time() - start_time
        mean_time = np.mean(total_time)
        # mean_time = curr_time / iteration
        mean_fps = 1 / mean_time

    print(f"the mean time is {mean_time:1.7f}, the mean fps is {mean_fps:1.7f}")


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser("Training")

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar-10-10', help="")
    parser.add_argument('--loss', type=str, default='ARPLoss', help='For cifar-10-100')

    args = parser.parse_args()

    for dataset in ('mnist', 'svhn', 'cifar-10-10', 'cifar-10-100', 'tinyimagenet'):
        args.dataset = dataset
        args = get_default_hyperparameters(args)
        print(f'{dataset}')
        print(args)
