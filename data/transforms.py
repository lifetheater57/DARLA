"""Data transforms for the loaders
"""
import torch
import torch.nn.functional as F
from torchvision import transforms as trsfs
import numpy as np
from PIL import Image
import traceback


def get_transform(transform_item):
    """Returns the torchivion transform function associated to a
    transform_item listed in opts.data.transforms ; transform_item is
    an addict.Dict
    """

    if transform_item.name == "crop" and not transform_item.ignore:
        return trsfs.RandomCrop((transform_item.height, transform_item.width))

    if transform_item.name == "resize" and not transform_item.ignore:
        return trsfs.Resize(
            transform_item.new_size  # , transform_item.get("keep_aspect_ratio", False)
        )

    if transform_item.name == "colorjitter" and not transform_item.ignore:
        return trsfs.ColorJitter(
            brightness=transform_item.brightness,
            contrast=transform_item.contrast,
            saturation=transform_item.saturation,
            hue=transform_item.hue,
        )

    if transform_item.name == "hflip" and not transform_item.ignore:
        return trsfs.RandomHorizontalFlip(p=transform_item.p or 0.5)

    if transform_item.ignore:
        return None

    raise ValueError("Unknown transform_item {}".format(transform_item))


def get_transforms(opts):
    """Get all the transform functions listed in opts.data.transforms
    using get_transform(transform_item)
    """
    last_transforms = [
        trsfs.ToTensor(),
        trsfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    conf_transforms = []
    for t in opts.data.transforms:
        if get_transform(t) is not None:
            conf_transforms.append(get_transform(t))

    return conf_transforms + last_transforms
