import os
import os.path as osp
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from tqdm import tqdm
from methods.ARPL.core import evaluation
from methods.ARPL.arpl_utils import AverageMeter

from sklearn.metrics import average_precision_score
import pandas as pd
from utils.tsne import visualize_tsne


def test(net, criterion, testloader, outloader, epoch=None, **options):
    net.eval()
    losses_known = AverageMeter()
    loss_all_known = 0
    losses_unknown = AverageMeter()
    loss_all_unknown = 0
    correct, total = 0, 0
    features_known = None
    features_unknown = None
    logits_known = None

    labels_known = []
    labels_unknown = []
    features_all = None
    labels_all = []
    known_cls_num = len(options['train_classes'])

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels, _labels_u = [], [], [], []

    with torch.no_grad():
        for data, labels, idx in tqdm(testloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            labels_known += labels
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                current_fea_known = x.cpu().numpy()
                if features_known is not None:
                    features_known = np.concatenate((features_known, current_fea_known))
                else:
                    features_known = current_fea_known
                logits, loss_known = criterion(x, y, labels)

                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
                losses_known.update(loss_known.item(), data.size(0))

                loss_all_known += losses_known.avg

                if options['use_softmax_in_eval']:
                    logits = torch.nn.Softmax(dim=-1)(logits)

                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels, idx) in enumerate(tqdm(outloader)):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            labels_unknown += (labels + known_cls_num)
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                current_fea_unknown = x.cpu().numpy()
                if features_unknown is not None:
                    features_unknown = np.concatenate((features_unknown, current_fea_unknown))
                else:
                    features_unknown = current_fea_unknown
                logits, loss_unknown = criterion(x, y, None)
                # losses_unknown.update(loss_unknown.item(), data.size(0))
                #
                # loss_all_unknown += losses_unknown.avg

                if options['use_softmax_in_eval']:
                    logits = torch.nn.Softmax(dim=-1)(logits)

                _pred_u.append(logits.data.cpu().numpy())
                _labels_u.append(labels.data.cpu().numpy())

    features_all = np.concatenate((features_known, features_unknown))
    labels_all = labels_known + labels_unknown
    all_classes = options['train_classes'] + options['open_set_classes']

    # sil_score, cal_score, dav_score = evaluation.clustering_metrics(features_known, options['train_classes'])
    # visualize_tsne(features_unknown, labels_unknown, options["expr_name"], epoch, "unknown")
    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)

    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']

    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    # Average precision
    ap_score = average_precision_score([0] * len(_pred_k) + [1] * len(_pred_u),
                                       list(-np.max(_pred_k, axis=-1)) + list(-np.max(_pred_u, axis=-1)))

    # confusion matrix
    df_cm = pd.DataFrame()
    if (epoch + 1) == options['max_epoch']:
        save_path = osp.join("ckpt", options["expr_name"])
        # visualize_tsne(features_all, labels_all, save_path, epoch, "all", all_classes, options['dataset'],
        #                unknown_in_one=True)
        visualize_tsne(features_known, labels_known, save_path, epoch, "known", options['train_classes'],
                       options['dataset'])
        df_cm = evaluation.get_confusion_matrix(_pred_k, _labels, options['train_classes'], options['dataset'])
    results['DF_CM'] = df_cm
    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.
    results['AUPR'] = ap_score * 100
    # results['Sil'] = sil_score
    # results['Cal'] = cal_score
    # results['Dav'] = dav_score

    return results, loss_all_known, loss_all_unknown


