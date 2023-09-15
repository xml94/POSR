import torch
import numpy as np
import os
import json

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, average_precision_score, \
    confusion_matrix, ConfusionMatrixDisplay, silhouette_score, calinski_harabasz_score, \
    davies_bouldin_score
from tqdm import tqdm
from data.open_set_splits.osr_splits import osr_splits
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from utils.tsne import visualize_tsne
from utils.confusion_matrix import pp_matrix
from sklearn.cluster import KMeans
from sklearn.metrics._ranking import _binary_clf_curve


def binary_confusion_matrix(y_true, y_pred, tpr):
    fps, tps, thresholds = _binary_clf_curve(y_true, y_pred,
                                             pos_label=None,
                                             sample_weight=None)
    fns = tps[-1] - tps
    tns = fps[-1] - fps
    fps_all, tps_all, fns_all, tns_all = [], [], [], []

    thre = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    for i in thre:
        _, idx = find_nearest(tpr, i)
        t = thresholds[idx]
        index = list(thresholds).index(t)
        fps_all.append(fps[index])
        tps_all.append(tps[index])
        fns_all.append(fns[index])
        tns_all.append(tns[index])

    return fps_all, tps_all, fns_all, tns_all


def normalised_average_precision(y_true, y_pred):
    fps, tps, thresholds = _binary_clf_curve(y_true, y_pred,
                                             pos_label=None,
                                             sample_weight=None)

    n_pos = np.array(y_true).sum()
    n_neg = (1 - np.array(y_true)).sum()

    precision = tps * n_pos / (tps * n_pos + fps * n_neg)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    precision, recall, thresholds = np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]

    return -np.sum(np.diff(recall) * np.array(precision)[:-1])


def find_nearest(array, value):
    array = np.asarray(array)
    length = len(array)
    abs_diff = np.abs(array - value)

    t_star = abs_diff.min()
    equal_arr = (abs_diff == t_star).astype('float32') + np.linspace(start=0, stop=0.1, num=length)

    idx = equal_arr.argmax()

    return array[idx], idx


def acc_at_t(preds, labels, t):
    pred_t = np.copy(preds)
    pred_t[pred_t > t] = 1
    pred_t[pred_t <= t] = 0

    acc = accuracy_score(labels, pred_t.astype('int32'))

    return acc


# def acc_at_t_curv(preds, labels, thresholds, tpr, save_path):
#     acc_all = []
#     t_all = []
#     thre = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99, 1.00]
#     for i in thre:
#         _, idx = find_nearest(tpr, i)
#         t = thresholds[idx]
#         t_all.append(t)
#     for t in t_all:
#         pred_t = np.copy(preds)
#         pred_t[pred_t > t] = 1
#         pred_t[pred_t <= t] = 0
#
#         acc = accuracy_score(labels, pred_t.astype('int32'))
#         acc_all.append(acc)
#
#     # plt.plot()
#     plt.xlabel("True positive rate")
#     plt.ylabel("open_set classification acc")
#     plt.plot(thre, acc_all)
#     save_path_ = save_path.replace('.svg', f"_acc_at_thresh_curve.svg")
#     plt.savefig(save_path_)
#     plt.clf()
def acc_at_t_curv(preds, labels, thresholds, tpr):
    acc_all = []
    t_all = []
    thre = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99, 1.00]
    for i in thre:
        _, idx = find_nearest(tpr, i)
        t = thresholds[idx]
        t_all.append(t)
    for t in t_all:
        pred_t = np.copy(preds)
        pred_t[pred_t > t] = 1
        pred_t[pred_t <= t] = 0

        acc = accuracy_score(labels, pred_t.astype('int32'))
        acc_all.append(acc)

    return acc_all


def closed_set_acc(preds, labels):
    preds = preds.argmax(axis=-1)
    acc = accuracy_score(labels, preds)

    print('Closed Set Accuracy: {:.3f}'.format(acc))

    return acc


def tar_at_far_and_reverse(fpr, tpr, thresholds):
    # TAR at FAR
    tar_at_far_all = {}
    for t in thresholds:
        tar_at_far_all[t] = None

    for t in thresholds:
        _, idx = find_nearest(fpr, t)
        tar_at_far = tpr[idx]
        tar_at_far_all[t] = tar_at_far

        print(f'TAR @ FAR {t}: {tar_at_far}')

    # FAR at TAR
    far_at_tar_all = {}
    for t in thresholds:
        far_at_tar_all[t] = None

    for t in thresholds:
        _, idx = find_nearest(tpr, t)
        far_at_tar = fpr[idx]
        far_at_tar_all[t] = far_at_tar

        print(f'FAR @ TAR {t}: {far_at_tar}')


def acc_at_95_tpr(open_set_preds, open_set_labels, thresholds, tpr):
    # Error rate at 95% TPR
    _, idx = find_nearest(tpr, 0.95)
    t = thresholds[idx]
    acc_at_95 = acc_at_t(open_set_preds, open_set_labels, t)
    print(f'Error Rate at TPR 95%: {1 - acc_at_95}')

    return acc_at_95, t


def compute_auroc(open_set_preds, open_set_labels):
    auroc = roc_auc_score(open_set_labels, open_set_preds)
    print(f'AUROC: {auroc}')

    return auroc


def compute_aupr(open_set_preds, open_set_labels, normalised_ap=False):
    if normalised_ap:
        aupr = normalised_average_precision(open_set_labels, open_set_preds)
    else:
        aupr = average_precision_score(open_set_labels, open_set_preds)
    print(f'AUPR: {aupr}')

    return aupr


def compute_oscr(x1, x2, pred, labels):
    """
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate
    """

    x1, x2 = -x1, -x2

    # x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    # pred = np.argmax(pred_k, axis=1)

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
    # print(f"s_k_target is {s_k_target}")
    # print(f"s_u_target is {s_u_target}")

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

    print(f'OSCR: {OSCR}')

    return OSCR


def compute_open_preds(x1, x2, pred, labels, open_labels, open_img_path):
    x1, x2 = -x1, -x2

    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)

    idx = predict.argsort()
    s_predict = predict[idx]

    s_k_target = k_target[idx]
    s_u_labels = open_labels[idx]
    s_u_img_path = open_img_path[idx]

    return s_k_target, s_predict, s_u_labels, s_u_img_path


def get_confusion_matrix(pred_k, labels_k, known_classes, dataset, save_path):
    class_name = get_class_name(dataset)
    known_class_name = [class_name[x] for x in known_classes]
    print(f"pred_k are {pred_k}")
    x1 = np.argmax(pred_k, axis=1)
    cm_k = confusion_matrix(labels_k, x1)

    df_cm = pd.DataFrame(cm_k, index=known_class_name,
                         columns=known_class_name)
    pp_matrix(df_cm, cmap='Blues', cbar=True)
    plt.plot()
    save_path_ = save_path.replace('.svg', "confusion_matrix.svg")
    plt.savefig(save_path_)
    plt.clf()


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


def clustering_metrics(features, n_classes):
    n_cluster = len(n_classes)
    labels = KMeans(n_clusters=n_cluster).fit(features).labels_
    sil_score = silhouette_score(features, labels)
    cal_score = calinski_harabasz_score(features, labels)
    dav_score = davies_bouldin_score(features, labels)
    return sil_score, cal_score, dav_score


class EvaluateOpenSet():

    def __init__(self, model, save_dir, known_data_loader, unknown_data_loader, device=None, split_idx=0,
                 known_class=None, unknown_class=None, dataset=None):

        self.model = model
        self.known_data_loader = known_data_loader
        self.unknown_data_loader = unknown_data_loader
        self.save_dir = save_dir
        self.split_idx = split_idx
        self.known_class = known_class
        self.unknown_class = unknown_class
        self.dataset = dataset

        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.device = device

        # Init empty lists for saving labels and preds
        self.closed_set_preds = {0: [], 1: []}
        self.open_set_preds = {0: [], 1: []}

        self.closed_set_labels = {0: [], 1: []}
        self.open_set_labels = {0: [], 1: []}

        self.closed_set_imgPath = {0: [], 1: []}
        self.open_set_imgPath = {0: [], 1: []}

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def predict(self):

        with torch.no_grad():
            for open_set_label, loader in enumerate((self.known_data_loader, self.unknown_data_loader)):

                if open_set_label:
                    print('Forward pass through Open Set test set...')
                else:
                    print('Forward pass through Closed Set test set...')

                for batch_idx, batch in enumerate(tqdm(loader)):
                    imgs, labels, idxs = [x for x in batch]
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    # img_path, cls_idx = [x for x in idxs]
                    # img_path = list(img_path)

                    # Model forward
                    output = self.model(imgs)
                    closed_set_preds, open_set_preds = [x.cpu().numpy().tolist() for x in output]

                    # Update preds and labels
                    self.closed_set_preds[open_set_label].extend(closed_set_preds)
                    self.open_set_preds[open_set_label].extend(open_set_preds)

                    self.closed_set_labels[open_set_label].extend(labels.cpu().numpy().tolist())
                    self.open_set_labels[open_set_label].extend([open_set_label] * len(labels))

                    # self.closed_set_imgPath[open_set_label].extend(img_path)
                    # self.open_set_imgPath[open_set_label].extend(img_path)

        # Save to disk
        save_names = ['closed_set_preds.pt', 'open_set_preds.pt', 'closed_set_labels.pt', 'open_set_labels.pt']
        save_lists = [self.closed_set_preds, self.open_set_preds, self.closed_set_labels, self.open_set_labels]

        for name, x in zip(save_names, save_lists):
            path = os.path.join(self.save_dir, name)
            torch.save(x, path)

        # openSetpreds = np.array(self.open_set_preds[0] + self.open_set_preds[1])
        # openSetlabels = np.array(self.open_set_labels[0] + self.open_set_labels[1])
        # openSetIMGPath = np.array(self.open_set_imgPath[0] + self.open_set_imgPath[1])
        # open_set_preds_known_cls = openSetpreds[~openSetlabels.astype('bool')]
        # open_set_preds_unknown_cls = openSetpreds[openSetlabels.astype('bool')]
        # pred_known, pred_unknown, unknown_labels, img_path = compute_open_preds(open_set_preds_known_cls,
        #                                                                         open_set_preds_unknown_cls,
        #                                                                         np.array(
        #                                                                             self.closed_set_preds[0]).argmax(
        #                                                                             axis=-1),
        #                                                                         self.closed_set_labels[0],
        #                                                                         openSetlabels, openSetIMGPath)
        #
        # save_names_json = ['closed_set_preds.json', 'open_set_preds_known.json', 'closed_set_labels.json',
        #                    'open_set_labels.json', 'open_set_preds_unknown.json',
        #                    'imgPath.json', 'known_imgPath.json']
        # save_lists_json = [np.array(self.closed_set_preds[0]).argmax(axis=-1).tolist(),
        #                    pred_known.tolist(), self.closed_set_labels[0], unknown_labels.tolist(),
        #                    pred_unknown.tolist(), img_path.tolist(), self.closed_set_imgPath[0]]
        #
        # for name, x in zip(save_names_json, save_lists_json):
        #     path = os.path.join(self.save_dir, name)
        #     with open(path, 'w') as file:
        #         json.dump(x, file, indent=4)

    @staticmethod
    def evaluate(self, load=True, preds=None, normalised_ap=False):

        if load:
            save_names = ['closed_set_preds.pt', 'open_set_preds.pt', 'closed_set_labels.pt', 'open_set_labels.pt']

            closed_set_preds, open_set_preds, closed_set_labels, open_set_labels = \
                [torch.load(os.path.join(self.save_dir, name)) for name in save_names]

        else:

            closed_set_preds, open_set_preds, closed_set_labels, open_set_labels = preds

        # concatenate known and unknown
        open_set_preds = np.array(open_set_preds[0] + open_set_preds[1])
        open_set_labels = np.array(open_set_labels[0] + open_set_labels[1])

        # ----------------------------
        # CLOSED SET EVALUATION
        # ----------------------------

        test_acc = closed_set_acc(np.array(closed_set_preds[0]), np.array(closed_set_labels[0]))

        # ----------------------------
        # OPEN SET EVALUATION
        # ----------------------------

        figure_file_name = os.path.join(self.save_dir, str(self.split_idx) + "figure.svg")
        fpr, tpr, thresh = roc_curve(open_set_labels, open_set_preds, drop_intermediate=False)
        acc_95, thresh_95 = acc_at_95_tpr(open_set_preds, open_set_labels, thresh, tpr)
        auroc = compute_auroc(open_set_preds, open_set_labels)
        aupr = compute_aupr(open_set_preds, open_set_labels, normalised_ap=normalised_ap)
        fps, tps, fns, tns = binary_confusion_matrix(open_set_labels, open_set_preds, tpr)
        print(f"fps is {fps}\n tps is {tps} \n fns is {fns} \n tns is {tns}")
        acc_all = acc_at_t_curv(open_set_preds, open_set_labels, thresh, tpr)

        # OSCR calcs
        open_set_preds_known_cls = open_set_preds[~open_set_labels.astype('bool')]
        open_set_preds_unknown_cls = open_set_preds[open_set_labels.astype('bool')]
        closed_set_preds_pred_cls = np.array(closed_set_preds[0]).argmax(axis=-1)
        labels_known_cls = np.array(closed_set_labels[0])
        open_set_labels_unknown_cls = open_set_labels[open_set_labels.astype('bool')]
        labels_unknown = np.array(closed_set_labels[1]) + len(self.known_class)
        labels_unknown = labels_unknown.tolist()
        feature_all = np.concatenate((np.array(closed_set_preds[0]), np.array(closed_set_preds[1])))
        label_all = np.array(closed_set_labels[0] + labels_unknown)
        all_classes = self.known_class + self.unknown_class

        oscr = compute_oscr(open_set_preds_known_cls, open_set_preds_unknown_cls, closed_set_preds_pred_cls,
                            labels_known_cls)
        # visualize_tsne(np.array(closed_set_preds[0]), labels_known_cls, self.save_dir, 99,
        #                "known" + str(self.split_idx), self.known_class, self.dataset)
        # visualize_tsne(feature_all, label_all, self.save_dir, 99, "all" + str(self.split_idx), all_classes,
        #                self.dataset, unknown_in_one=False)
        # get_confusion_matrix(feature_all, label_all, all_classes, self.dataset,
        #                      figure_file_name)
        # get_confusion_matrix(np.array(closed_set_preds[0]), labels_known_cls, self.known_class, self.dataset,
        #                      figure_file_name)
        sil_score, cal_score, dav_score = clustering_metrics(np.array(closed_set_preds[0]), self.known_class)
        print(f"sil_score are {sil_score}, cal_score are {cal_score}, dav_score are {dav_score}")

        return (test_acc, acc_95, auroc, aupr, oscr, sil_score, cal_score, dav_score, fps, tps, fns, tns, acc_all)
        # return (test_acc, acc_95, auroc, aupr, oscr, fps, tps, fns, tns, acc_all)


class EvaluateOpenSetInline(EvaluateOpenSet):

    def __init__(self, *args, **kwargs):

        super(EvaluateOpenSetInline, self).__init__(*args, **kwargs)

    def predict_and_eval(self):

        self.model.eval()

        print('Testing Open Set...')

        with torch.no_grad():
            for open_set_label, loader in enumerate((self.known_data_loader, self.unknown_data_loader)):
                for batch_idx, batch in enumerate(tqdm(loader)):
                    imgs, labels, idxs = [x.to(self.device) for x in batch]

                    # Model forward
                    output = self.model(imgs)
                    closed_set_preds, open_set_preds = [x.cpu().numpy().tolist() for x in output]

                    # Update preds and labels
                    self.closed_set_preds[open_set_label].extend(closed_set_preds)
                    self.open_set_preds[open_set_label].extend(open_set_preds)

                    self.closed_set_labels[open_set_label].extend(labels.cpu().numpy().tolist())
                    self.open_set_labels[open_set_label].extend([open_set_label] * len(labels))

        open_set_preds = np.array(self.open_set_preds[0] + self.open_set_preds[1])
        open_set_labels = np.array(self.open_set_labels[0] + self.open_set_labels[1])

        # ----------------------------
        # CLOSED SET EVALUATION
        # ----------------------------

        test_acc = closed_set_acc(np.array(self.closed_set_preds[0]), np.array(self.closed_set_labels[0]))

        # ----------------------------
        # OPEN SET EVALUATION
        # ----------------------------

        fpr, tpr, thresh = roc_curve(open_set_labels, open_set_preds, drop_intermediate=False)
        acc_95 = acc_at_95_tpr(open_set_preds, open_set_labels, thresh, tpr)
        auroc = compute_auroc(open_set_preds, open_set_labels)

        return (test_acc, acc_95, auroc)


class ModelTemplate(torch.nn.Module):

    def forward(self, imgs):
        """
        :param imgs:
        :return: Closed set and open set predictions on imgs
        """
        pass


if __name__ == '__main__':
    from sklearn.metrics.ranking import precision_recall_curve

    np.random.seed(0)

    y_true = [0] * 40 + [1] * 60
    y_pred = np.random.uniform(size=(100,))


    def _binary_uninterpolated_average_precision(
            y_true, y_score):
        precision, recall, _ = precision_recall_curve(
            y_true, y_score, None, None)
        # Return the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])


    ap = average_precision_score(y_true, y_pred)
    ap1 = _binary_uninterpolated_average_precision(y_true, y_pred)
    ap2 = normalised_average_precision(y_true, y_pred)

    print(ap)
    print(ap1)
    print(ap2)
