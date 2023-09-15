from models.classifier32 import classifier32
from methods.ARPL.arpl_models.resnetABN import resnet50ABN
from methods.ARPL.arpl_models.arpl_models import classifier32ABN
from methods.ARPL.loss.ARPLoss import ARPLoss
from functools import partial
from utils.utils import strip_state_dict
from utils.pos_embed import interpolate_pos_embed
from config import imagenet_moco_path, places_supervised_path, places_moco_path, imagenet_supervised_path, \
    coco_moco_path, imagenet_mae_base_path, imagenet_mae_large_path, vit_in1k_mae_plantclef_softmax_large_path, \
    vit_in1k_supervised_large_path, vit_in21k_supervised_large_path
from methods.ARPL.arpl_models import models_vit, models_resnet, models_vit_org
from timm.models.layers import trunc_normal_
import timm
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F


class TimmResNetWrapper(torch.nn.Module):

    def __init__(self, resnet, admloss):

        super().__init__()
        self.resnet = resnet
        self.admloss = admloss

    def forward(self, x, return_features=True, dummy_label=None):

        x = self.resnet.forward_features(x)
        embedding = self.resnet.global_pool(x)
        if self.resnet.drop_rate:
            embedding = torch.nn.functional.dropout(embedding, p=float(self.drop_rate), training=self.training)
        if self.admloss:
            F.normalize(self.resnet.fc.weight)
            embedding = F.normalize(embedding, dim=1)
        preds = self.resnet.fc(embedding)

        if return_features:
            return embedding, preds
        else:
            return preds


class TimmViTWrapper(nn.Module):

    def __init__(self, vit, admloss):

        super().__init__()
        self.vit = vit
        self.admloss = admloss

    def forward(self, x, return_features=True, dummy_label=None):

        x = self.vit.forward_features(x)
        x = self.vit.forward_head(x)

        if return_features:
            return x, x
        else:
            return x


class Classifier32ARPLWrapper(torch.nn.Module):

    def __init__(self, base_model, loss_layer):

        super().__init__()

        self.base_model = base_model
        self.loss_layer = loss_layer

    def forward(self, imgs, return_feature=False):

        x, y = self.base_model(imgs, True)
        logits, _ = self.loss_layer(x, y)

        if return_feature:
            return x, logits
        else:
            return logits

    def load_state_dict(self, state_dict):

        """
        Override method to take list of state dicts for loss layer and criterion
        """

        base_model_state_dict, loss_layer_state_dict = [strip_state_dict(s) for s in state_dict]
        # base_model_state_dict = strip_state_dict(base_model_state_dict, strip_key='base_model.')

        self.base_model.load_state_dict(base_model_state_dict)
        self.loss_layer.load_state_dict(loss_layer_state_dict)

        self.base_model.eval()
        self.loss_layer.eval()


class VitARPLWrapper(torch.nn.Module):

    def __init__(self, base_model, loss_layer):

        super().__init__()

        self.base_model = base_model
        self.loss_layer = loss_layer

    def forward(self, imgs):

        x, y = self.base_model(imgs)
        logits, _ = self.loss_layer(x, y)

        return logits

    def load_state_dict(self, state_dict):

        """
        Override method to take list of state dicts for loss layer and criterion
        """

        base_model_state_dict, loss_layer_state_dict = [strip_state_dict(s) for s in state_dict]
        # base_model_state_dict = strip_state_dict(base_model_state_dict, strip_key='base_model.')

        self.base_model.load_state_dict(base_model_state_dict)
        self.loss_layer.load_state_dict(loss_layer_state_dict)

        self.base_model.eval()
        self.loss_layer.eval()


def transform_moco_state_dict_places(obj, num_classes, supervised=False):
    """
    Transforms state dict from Places pretraining here: https://github.com/nanxuanzhao/Good_transfer
    :param obj: Moco State Dict
    :param args: argsparse object with training classes
    :return: State dict compatable with standard resnet architecture
    """

    if supervised:

        newmodel = obj
        newmodel['fc.weight'] = torch.randn((num_classes, 2048))
        newmodel['fc.bias'] = torch.randn((num_classes,))

    else:

        newmodel = {}
        for k, v in obj.items():

            if k.startswith("fc.2"):
                continue

            if k.startswith("fc.0"):
                k = k.replace("0.", "")
                if "weight" in k:
                    v = torch.randn((num_classes, v.size(1)))
                elif "bias" in k:
                    v = torch.randn((num_classes,))

            newmodel[k] = v

    return newmodel


def transform_state_dict_plantclef(obj, num_classes):
    """
    Transforms state dict from PlantCLEF pretraining
    :param obj: State Dict
    :param args: argsparse object with training classes
    :return: State dict compatable with standard resnet architecture
    """

    newmodel = obj['model']
    newmodel['fc.weight'] = torch.randn((num_classes, 2048))
    newmodel['fc.bias'] = torch.randn((num_classes,))

    return newmodel


def transform_moco_state_dict(obj, num_classes):
    """
    :param obj: Moco State Dict
    :param args: argsparse object with training classes
    :return: State dict compatable with standard resnet architecture
    """

    newmodel = {}
    for k, v in obj.items():
        if not k.startswith("module.encoder_q."):
            continue
        old_k = k
        k = k.replace("module.encoder_q.", "")

        if k.startswith("fc.2"):
            continue

        if k.startswith("fc.0"):
            k = k.replace("0.", "")
            if "weight" in k:
                v = torch.randn((num_classes, v.size(1)))
            elif "bias" in k:
                v = torch.randn((num_classes,))

        newmodel[k] = v

    return newmodel


def transform_moco_state_dict_arpl_cs(obj, num_classes):
    """
    :param obj: Moco State Dict
    :param args: argsparse object with training classes
    :return: State dict compatable with standard resnet architecture
    """

    newmodel = {}
    for k, v in obj.items():
        if not k.startswith("module.encoder_q."):
            continue
        old_k = k
        k = k.replace("module.encoder_q.", "")

        if k.startswith("fc.2"):
            continue

        if k.startswith("fc.0"):
            k = k.replace("0.", "")
            if "weight" in k:
                v = torch.randn((num_classes, v.size(1)))
            elif "bias" in k:
                v = torch.randn((num_classes,))

        newmodel[k] = v

    # For newmodel, change all batch norms from bnX.XXX --> bnX.bns.0.XXX
    #                                                   add bnX.bns.1.XXX with same params

    newmodel2 = {}
    for k, v in newmodel.items():

        if 'bn' in k:
            parts = k.split('.')
            if k.startswith('bn1'):

                newk1 = '.'.join([parts[0], 'bns', '0', parts[-1]])
                newk2 = '.'.join([parts[0], 'bns', '1', parts[-1]])

            else:

                idx = [i for i, x in enumerate(parts) if 'bn' in x]
                idx = idx[0] + 1
                newk1 = '.'.join([*parts[:idx], 'bns', '0', *parts[idx:]])
                newk2 = '.'.join([*parts[:idx], 'bns', '1', *parts[idx:]])

            newmodel2[newk1] = v
            newmodel2[newk2] = v


        elif 'downsample' in k:

            if 'downsample.0' in k:

                newmodel2[k] = v

            else:

                parts = k.split('.')
                idx = len(parts) - 1

                newk1 = '.'.join([*parts[:idx], 'bns', '0', *parts[idx:]])
                newk2 = '.'.join([*parts[:idx], 'bns', '1', *parts[idx:]])

                newmodel2[newk1] = v
                newmodel2[newk2] = v

        else:

            newmodel2[k] = v

    return newmodel2


def get_model(args, wrapper_class=None, evaluate=False, *args_, **kwargs):
    if args.model == 'resnet50':

        # Get model
        if args.cs:
            model = resnet50ABN(num_classes=len(args.train_classes), num_bns=2, first_layer_conv=7)
        else:
            # model = timm.create_model(args.model, num_classes=len(args.train_classes))
            model = models_resnet.__dict__[args.model](
                num_classes=len(args.train_classes),
                admloss=args.admloss,
            )
        if args.pretrain != '':
            # Get function to transform state_dict and state_dict path
            if args.pretrain == 'imagenet_moco':
                pretrain_path = imagenet_moco_path
                state_dict_transform = partial(transform_moco_state_dict_places, supervised=False)  # add by mengyao
            elif args.pretrain == 'imagenet':
                pretrain_path = imagenet_supervised_path
                state_dict_transform = partial(transform_moco_state_dict_places, supervised=True)
            elif args.pretrain == 'imagenet_in21k':
                pretrain_path = imagenet_miil_in21k_path
                state_dict_transform = partial(transform_moco_state_dict_places, supervised=True)
            elif args.pretrain == 'places_moco':
                pretrain_path = places_moco_path
                state_dict_transform = partial(transform_moco_state_dict_places, supervised=False)
            elif args.pretrain == 'coco_moco':
                pretrain_path = coco_moco_path
                state_dict_transform = partial(transform_moco_state_dict_places, supervised=False)
            elif args.pretrain == 'places':
                pretrain_path = places_supervised_path
                state_dict_transform = partial(transform_moco_state_dict_places, supervised=True)
            elif args.pretrain == 'cnn_plantclef_softmax':
                pretrain_path = cnn_plantclef_softmax_path
                state_dict_transform = partial(transform_state_dict_plantclef)
            elif args.pretrain == 'cnn_in1k_softmax_plantclef_softmax':
                pretrain_path = cnn_in1k_softmax_plantclef_softmax_path
                state_dict_transform = partial(transform_state_dict_plantclef)
            elif args.pretrain == 'cnn_in1k_moco_plantclef_softmax':
                pretrain_path = cnn_in1k_moco_plantclef_softmax_path
                state_dict_transform = partial(transform_state_dict_plantclef)
            elif args.pretrain == 'imagenet_moco' and args.cs:
                pretrain_path = imagenet_moco_path
                state_dict_transform = transform_moco_state_dict_arpl_cs  # Note, not implemented for imagenet pretraining
            else:
                raise NotImplementedError

            state_dict = torch.load(pretrain_path)  # add by mengyao stand for upper line code
            if args.pretrain == 'imagenet_in21k':
                state_dict = state_dict['state_dict']
            state_dict = strip_state_dict(state_dict, strip_key='module.')
            state_dict = state_dict_transform(state_dict, len(args.train_classes))

            model.load_state_dict(state_dict)

        # If loss is ARPLoss, bolt on loss layer to model
        if args.loss == 'ARPLoss':
            if evaluate:
                model = TimmResNetWrapper(model, args.admloss)

                loss_layer = ARPLoss(use_gpu=True, weight_pl=0.0, temp=1, num_classes=len(args.train_classes),
                                     feat_dim=2048, label_smoothing=0.9, admloss=args.admloss)

                model = Classifier32ARPLWrapper(base_model=model, loss_layer=loss_layer)

    elif args.model == 'classifier32':

        try:
            feat_dim = args.feat_dim
            cs = args.cs
        except:
            feat_dim = None
            cs = False

        model = classifier32(num_classes=len(args.train_classes), feat_dim=feat_dim)

        if args.loss == 'ARPLoss':
            if evaluate:
                if cs:
                    model = classifier32ABN(feat_dim=feat_dim, num_classes=len(args.train_classes))

                    loss_layer = ARPLoss(use_gpu=True, weight_pl=0.0, temp=1, num_classes=len(args.train_classes),
                                         feat_dim=args.feat_dim, label_smoothing=0.9, admloss=args.admloss)

                    model = Classifier32ARPLWrapper(base_model=model, loss_layer=loss_layer)

    elif args.model in ['wide_resnet50_2', 'efficientnet_b0', 'efficientnet_b7', 'dpn92']:

        model = timm.create_model(args.model, num_classes=len(args.train_classes))

    elif args.model == 'vit_large_patch16':
        if args.pretrain == 'vit_in1k_supervised_large' or args.pretrain == 'vit_in21k_supervised_large':
            model = models_vit_org.__dict__[args.model](
                num_classes=args.nb_classes,
                drop_path_rate=args.drop_path,
                global_pool=args.global_pool,
                admloss=args.admloss,
            )
            print(f"pretrain is {args.pretrain}")
            if not evaluate:
                pretrain_path = None
                if args.pretrain == 'vit_in1k_supervised_large':
                    pretrain_path = vit_in1k_supervised_large_path
                elif args.pretrain == 'vit_in21k_supervised_large':
                    pretrain_path = vit_in21k_supervised_large_path
                print("Load pre-trained checkpoint from: %s" % pretrain_path)
                timm.models.vision_transformer._load_weights(model, pretrain_path)
                # model.load_pretrained(pretrain_path)

        else:
            model = models_vit.__dict__[args.model](
                num_classes=args.nb_classes,
                drop_path_rate=args.drop_path,
                global_pool=args.global_pool,
                admloss=args.admloss,
            )
            print(f"pretrain is {args.pretrain}")
            if args.pretrain != '':
                pretrain_path = None
                if args.pretrain == 'imagenet_large':
                    pretrain_path = imagenet_mae_large_path
                elif args.pretrain == 'vit_in1k_mae_plantclef_softmax_large':
                    pretrain_path = vit_in1k_mae_plantclef_softmax_large_path

                print("Load pre-trained checkpoint from: %s" % pretrain_path)

                checkpoint = torch.load(pretrain_path, map_location='cpu')
                checkpoint_model = checkpoint['model']
                state_dict = model.state_dict()
                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]
                # interpolate position embedding
                interpolate_pos_embed(model, checkpoint_model)

                # load pre-trained model
                msg = model.load_state_dict(checkpoint_model, strict=False)
                print(msg)
                trunc_normal_(model.head.weight, std=2e-5)
        # If loss is ARPLoss, bolt on loss layer to model
        if args.loss == 'ARPLoss':
            if evaluate:
                model = TimmViTWrapper(model, args.admloss)

                loss_layer = ARPLoss(use_gpu=True, weight_pl=0.0, temp=1, num_classes=len(args.train_classes),
                                     feat_dim=1000, label_smoothing=0.9, admloss=args.admloss)

                model = VitARPLWrapper(base_model=model, loss_layer=loss_layer)

    else:

        raise NotImplementedError

    if wrapper_class is not None:
        model = wrapper_class(model, args.admloss, *args_, **kwargs)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='cls',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General
    parser.add_argument('--model', default='resnet50', type=str)
    parser.add_argument('--pretrain', type=str, default='places_moco',
                        help='Which pretraining to use if --model=timm_resnet50_pretrained.'
                             'Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')
    parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)
    parser.add_argument('--loss', type=str, default='ARPLoss')
    args = parser.parse_args()

    args.train_classes = (0, 1, 8, 9)
    model = get_model(args)
    x, y = model(torch.randn(64, 3, 32, 32), True)
    debug = True
