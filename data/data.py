from pathlib import Path
import yaml
import json
import torch
from torch.utils.data import DataLoader, Dataset
from imageio import imread
from torchvision import transforms
import numpy as np
from .transforms import get_transforms
from PIL import Image
import os


class DuckieDataset(Dataset):
    def __init__(self, mode, opts, transform=None):
        """[summary]

        Args:
            mode (str): train/test
            opts (config): 
            transform ([type], optional): [description]. Defaults to None.

        Raises:
            ValueError: [description]
        """
        file_list_path = Path(opts.data.files[mode])
        if "/" not in str(file_list_path):  # it is not an absolute path
            file_list_path = Path(opts.data.files.base) / Path(opts.data.files[mode])

        if file_list_path.suffix == ".json":
            self.samples_paths = self.json_load(file_list_path)
        elif file_list_path.suffix in {".yaml", ".yml"}:
            self.samples_paths = self.yaml_load(file_list_path)
        elif file_list_path.suffix in {".txt"}:
            self.samples_paths = self.txt_load(file_list_path)
        else:
            raise ValueError("Unknown file list type in {}".format(file_list_path))

        if opts.data.max_samples and opts.data.max_samples != -1:
            assert isinstance(opts.data.max_samples, int)
            self.samples_paths = self.samples_paths[: opts.data.max_samples]

        if opts.data.check_samples:
            print(f"Checking samples ({mode}, {domain})")
            self.check_samples()
        self.file_list_path = str(file_list_path)
        self.transform = transform

    def __getitem__(self, i):
        """Return an item in the dataset with fields:
        {
            data: transform({
                domains: values
            }),
            paths: [paths],
            mode: [train|val]
        }
        Args:
            i (int): index of item to retrieve
        Returns:
            dict: dataset item where tensors of data are in item["data"] which is a dict
                  {task: tensor}
        """
        path = self.samples_paths[i]

        # always apply transforms,
        # if no transform is specified, ToTensor and Normalize will be applied

        item = Image.open(path)
        return self.transform(item)

    def __len__(self):
        return len(self.samples_paths)

    def json_load(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def yaml_load(self, file_path):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    def txt_load(self, file_path):
        with open(file_path, "r") as f:
            return f.read().splitlines()

    def check_samples(self):
        """Checks that every file listed in samples_paths actually
        exist on the file-system
        """
        for s in self.samples_paths:
            assert Path(s).exists(), f"{s} does not exist"


def get_loader(opts, mode):
    return DataLoader(
        DuckieDataset(mode, opts, transform=transforms.Compose(get_transforms(opts))),
        batch_size=opts.data.loaders.get("batch_size", 4),
        shuffle=True,
        num_workers=opts.data.loaders.get("num_workers", 4),
        pin_memory=True,  # faster transfer to gpu
        drop_last=True,  # avoids batchnorm pbs if last batch has size 1
    )
