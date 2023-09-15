# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.resnet


class ResNet(timm.models.resnet.ResNet):
    def __init__(self, admloss=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.admloss = admloss

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        if self.admloss:
            F.normalize(self.fc.weight)
            x = F.normalize(x, dim=1)
        return x if pre_logits else self.fc(x)


def resnet50(**kwargs):
    model=ResNet(block=timm.models.resnet.Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return model
