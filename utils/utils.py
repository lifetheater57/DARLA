import os
import shutil
from addict import Dict
from pathlib import Path

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
