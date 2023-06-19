"""
For training from a pre-embedded dataset of image-text pairs from a clip model


Make sure that the CLIP model you use is the same as the one used to obtain the
pre embeddings
"""

import torch.utils.data
from math import ceil
from typing import List
import lightning.pytorch as pl
import numpy as np
from glob import glob
import re
import os

from torch.utils.data import DataLoader, Dataset, IterableDataset


def get_file_code(filename: str) -> int:
    fn = os.path.basename(filename)
    pattern = r"(\d+)(?=\.npy$)"
    m = re.search(pattern, fn)
    return int(m[0])


def random_sort(*lists):
    indices = np.random.permutation(len(lists[0]))
    return [l[i] for i in indices for l in lists]


class IterableImageTextPairDataset(IterableDataset):
    def __init__(self, path: str, batch_size: int, key_name: str):
        self.all_data_files = glob(path + "/*.npz")
        assert len(self.all_data_files) > 0
        self.key_name = key_name
        self.batch_size = batch_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, end = 0, len(self.all_data_files)
        else:
            n_files_per_worker = len(self.all_data_files) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * n_files_per_worker
            end = min(start + n_files_per_worker, len(self.all_data_files))
        return ImageTextPairDatasetWorker(
            self.all_data_files[start:end],
            batch_size=self.batch_size,
            key_name=self.key_name,
        )


class ImageTextPairDatasetWorker(IterableDataset):
    """
    key_name the first part of the accessor key, if the img embeddings are
    called: b32_img, the key_name is b32
    """

    def __init__(self, data_files: List[str], batch_size: int, key_name: str):
        self.key_name = key_name
        self.data_files = data_files

        # increasing this number improves random sampling
        self.num_files_to_load = 1

        self.batch_size = batch_size

        self.file_i = 0
        self.batch_i = 0
        self.__iter_file()

    def __iter_file(self):
        num_files_to_load = min(
            len(self.data_files) - self.file_i, self.num_files_to_load
        )
        print("__iter_file loading files ", num_files_to_load)
        text_data = []
        image_data = []
        n_loaded = 0
        while n_loaded < num_files_to_load:
            print("Loading file", self.data_files[self.file_i])
            try:
                dat = np.load(self.data_files[self.file_i])
                text_data.append(dat[self.key_name + "_txt"])
                image_data.append(dat[self.key_name + "_img"])
                n_loaded += 1
                assert len(text_data[-1]) == len(image_data[-1])
            except Exception as e:
                print("error loading file", self.data_files[self.file_i], e)
            self.file_i += 1

        text_data = np.concatenate(text_data, axis=0)
        image_data = np.concatenate(image_data, axis=0)

        rnd_indices = np.random.permutation(len(text_data))
        text_data = text_data[rnd_indices]
        image_data = image_data[rnd_indices]

        self.text_data = np.array_split(
            text_data, ceil(len(text_data) / self.batch_size)
        )
        self.image_data = np.array_split(
            image_data, ceil(len(image_data) / self.batch_size)
        )
        assert len(self.text_data) == len(self.image_data)
        self.batch_i = 0

    def __iter__(self):
        return self

    def __len__(self):
        n_files = len(self.data_files)
        num_rows_per_file = 500000
        return n_files * num_rows_per_file // self.batch_size

    def __next__(self):
        if self.batch_i >= len(self.image_data):
            if self.file_i >= len(self.data_files):
                raise StopIteration
            else:
                self.__iter_file()
        _ret = self.image_data[self.batch_i], self.text_data[self.batch_i]
        self.batch_i += 1
        return _ret


class LightningEmbeddingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path_train: str,
        path_val: str,
        batch_size: int,
        key_name: str,
        *_,
        **kwargs
    ):
        """
        path_train: some path to a directory with the following subfolders:
            images/*.npy
            texts/*.npy

        This module will iterate over all rows of all npy files.
        """
        super().__init__()
        self.batch_size = batch_size

        self.ds_train = IterableImageTextPairDataset(path_train, batch_size, key_name)
        self.ds_test = IterableImageTextPairDataset(path_val, batch_size, key_name)

    def train_dataloader(self):
        return DataLoader(self.ds_train, num_workers=4, batch_size=None, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.ds_test, num_workers=1, batch_size=None, shuffle=False)
