import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, silhouette_score, calinski_harabasz_score, \
    davies_bouldin_score
from sklearn.cluster import KMeans
from data.open_set_datasets import get_class_splits
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from utils.confusion_matrix import pp_matrix


def get_curve_online(known, novel, stypes=['Bas']):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known), np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        fp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k + num_n):
            if k == num_k:
                tp[stype][l + 1:] = tp[stype][l]
                fp[stype][l + 1:] = np.arange(fp[stype][l] - 1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l + 1:] = np.arange(tp[stype][l] - 1, -1, -1)
                fp[stype][l + 1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l + 1] = tp[stype][l]
                    fp[stype][l + 1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l + 1] = tp[stype][l] - 1
                    fp[stype][l + 1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95


def metric_ood(x1, x2, stypes=['Bas'], verbose=True):
    tp, fp, tnr_at_tpr95 = get_curve_online(x1, x2, stypes)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')

    for stype in stypes:
        if verbose:
            print('{stype:5s} '.format(stype=stype), end='')
        results[stype] = dict()

        # TNR
        mtype = 'TNR'
        results[stype][mtype] = 100. * tnr_at_tpr95[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')

        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype] / tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype] / fp[stype][0], [0.]])
        results[stype][mtype] = 100. * (-np.trapz(1. - fpr, tpr))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')

        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = 100. * (.5 * (tp[stype] / tp[stype][0] + 1. - fp[stype] / fp[stype][0]).max())
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')

        # AUIN
        mtype = 'AUIN'
        denom = tp[stype] + fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype] / denom, [0.]])
        results[stype][mtype] = 100. * (-np.trapz(pin[pin_ind], tpr[pin_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')

        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0] - tp[stype] + fp[stype][0] - fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0] - fp[stype]) / denom, [.5]])
        results[stype][mtype] = 100. * (np.trapz(pout[pout_ind], 1. - fpr[pout_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
            print('')

    return results


def compute_oscr(pred_k, pred_u, labels):
    x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    pred = np.argmax(pred_k, axis=1)
    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    return OSCR


def get_confusion_matrix(pred_k, labels_k, known_classes, dataset):

    class_name = get_class_name(dataset)
    known_class_name = [class_name[x] for x in known_classes]

    x1 = np.argmax(pred_k, axis=1)
    cm_k = confusion_matrix(labels_k, x1)

    df_cm = pd.DataFrame(cm_k, index=known_class_name,
                         columns=known_class_name)
    pp_matrix(df_cm, cmap='Blues', cbar=True)

    return df_cm


def get_class_name(dataset):
    class_name = []
    if dataset == 'ivadl_tomato':
        class_name = ['ulcer', 'leaf_fungus', 'septoria_spot',
                  'chlorosis_virus', 'Yellow_Curl', 'powdery_mildew',
                  'Healthy', 'leaf_miner', 'blueworms']
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


def clustering_metrics(features, n_classes):
    n_cluster = len(n_classes)
    labels = KMeans(n_clusters=n_cluster, n_init='auto').fit(features).labels_
    sil_score = silhouette_score(features, labels)
    cal_score = calinski_harabasz_score(features, labels)
    dav_score = davies_bouldin_score(features, labels)
    return sil_score, cal_score, dav_score




