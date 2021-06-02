from __future__ import absolute_import

from .resnet import *

__factory = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'ibn_resnet50': ibn_resnet50,
    'se_resnet50': se_resnet50,
    'se_resnext50': se_resnext50
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    :param name: str
    """
    if name not in __factory:
        return KeyError('Unknown models: ', name)
    return __factory[name](*args, **kwargs)