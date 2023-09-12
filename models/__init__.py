from __future__ import absolute_import
import torch
from .build_optimizer import build_optimizer

from .osnet_ain_geo import *

__model_factory = {
    # lightweight models
    'osnet_ain_x0_25_geo':osnet_ain_x0_25_geo,
    'osnet_ain_x0_75_geo':osnet_ain_x0_75_geo,
}


def show_avai_models():
    """Displays available models.
    """
    print(list(__model_factory.keys()))


def build_model(
    cfg, num_classes
):
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module
    """
    name=cfg.Method
    loss=cfg.loss
    pretrained=cfg.pretrained
    dropout=cfg.droprate
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    return __model_factory[name](
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=True,
        dropout_p=dropout
    )
