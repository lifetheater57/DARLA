import os
import shutil
from addict import Dict
from pathlib import Path
import numpy as np
import PIL
import torch
from torchvision import transforms

# utils files borrowed from https://github.com/cc-ai/omnigan/blob/master/omnigan/utils.py


def flatten_opts(opts: Dict) -> dict:
    """Flattens a multi-level addict.Dict or native dictionnary into a single
    level native dict with string keys representing the keys sequence to reach
    a value in the original argument.
    d = addict.Dict()
    d.a.b.c = 2
    d.a.b.d = 3
    d.a.e = 4
    d.f = 5
    flatten_opts(d)
    >>> {
        "a.b.c": 2,
        "a.b.d": 3,
        "a.e": 4,
        "f": 5,
    }
    Args:
        opts (addict.Dict or dict): addict dictionnary to flatten
    Returns:
        dict: flattened dictionnary
    """
    values_list = []

    def p(d, prefix="", vals=[]):
        for k, v in d.items():
            if isinstance(v, (Dict, dict)):
                p(v, prefix + k + ".", vals)
            elif isinstance(v, list):
                if v and isinstance(v[0], (Dict, dict)):
                    for i, m in enumerate(v):
                        p(m, prefix + k + "." + str(i) + ".", vals)
                else:
                    vals.append((prefix + k, str(v)))
            else:
                if isinstance(v, Path):
                    v = str(v)
                vals.append((prefix + k, v))

    p(opts, vals=values_list)
    return dict(values_list)


def env_to_path(path: str) -> str:
    """Transorms an environment variable mention in a json
    into its actual value. E.g. $HOME/clouds -> /home/user/clouds
    Args:
        path (str): path potentially containing the env variable
    """
    path_elements = path.split("/")
    new_path = []
    for el in path_elements:
        if "$" in el:
            new_path.append(os.environ[el.replace("$", "")])
        else:
            new_path.append(el)
    return "/".join(new_path)


def copy_run_files(opts: Dict) -> None:
    """
    Copy the opts's sbatch_file to output_path
    Args:
        opts (addict.Dict): options
    """
    if opts.sbatch_file:
        p = Path(opts.sbatch_file)
        if p.exists():
            o = Path(opts.output_path)
            if o.exists():
                shutil.copyfile(p, o / p.name)
    if opts.exp_file:
        p = Path(opts.exp_file)
        if p.exists():
            o = Path(opts.output_path)
            if o.exists():
                shutil.copyfile(p, o / p.name)

def tensor_to_PIL(image_tensor):
    image_array = image_tensor.cpu().numpy()
    if len(image_array.shape) == 4:
        image_array = image_array[0]
    image_array = image_array.transpose(1, 2, 0)
    image_array = (image_array * 255).astype(np.uint8)
    image = PIL.Image.fromarray(image_array)
    return image_array, image

def apply_random_mask(img):
    """Blank a rectangular region of random dimensions in the image.

    Args:
        img (tensor): The image on which to apply the mask.
    """

    img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]

    h_values = torch.empty(2).uniform_(0, img_h)
    w_values = torch.empty(2).uniform_(0, img_w)

    x = h_values.min().type(torch.IntTensor)
    y = w_values.min().type(torch.IntTensor)

    h = torch.abs(h_values[1] - h_values[0]).type(torch.IntTensor)
    w = torch.abs(w_values[1] - w_values[0]).type(torch.IntTensor)

    noise = torch.empty((img_c, h, w)).uniform_()

    return transforms.functional.erase(img, x, y, h, w, noise)