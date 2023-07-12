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


def get_file_code(filename: str):
    return os.path.basename(filename).split(".")[-2]


def random_sort(*lists):
    indices = np.random.permutation(len(lists[0]))
    return [l[i] for i in indices for l in lists]


class IterableImageTextPairDataset(IterableDataset):
    def __init__(self, path: str, batch_size: int, ):
        self.img_files = glob(path + "/image/*.npy")
        self.img_files.sort(key=get_file_code)

        self.txt_files = glob(path + "/text/*.npy")
        self.txt_files.sort(key=get_file_code)

        assert len(self.img_files) > 0
        assert len(self.img_files) == len(self.txt_files)

        for txt_file, img_file in zip(self.txt_files, self.img_files):
                assert os.path.basename(txt_file) == os.path.basename(img_file), f'{txt_file} != {img_file}'

        self.batch_size = batch_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, end = 0, len(self.img_files)
        else:
            n_files_per_worker = len(self.img_files) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * n_files_per_worker
            end = min(start + n_files_per_worker, len(self.img_files))
        return ImageTextPairDatasetWorker(
            self.img_files[start:end],
            self.txt_files[start:end],
            batch_size=self.batch_size,
        )


class ImageTextPairDatasetWorker(IterableDataset):
    def __init__(self, img_files: List[str], txt_files: List[str], batch_size: int, ):
        self.img_files = img_files
        self.txt_files = txt_files
        assert len(self.img_files) == len(self.txt_files)

        # increasing this number improves random sampling
        self.num_files_to_load = 2

        self.batch_size = batch_size

        self.file_i = 0
        self.batch_i = 0
        self.__iter_file()

    def __iter_file(self):
        num_files_to_load = min(
            len(self.txt_files) - self.file_i, self.num_files_to_load
        )
        print("__iter_file loading files ", num_files_to_load)
        text_data = []
        image_data = []
        n_loaded = 0
        while n_loaded < num_files_to_load:
            print("Loading files", self.txt_files[self.file_i], self.img_files[self.file_i])
            try:
                img_dat = np.load(self.img_files[self.file_i])
                txt_dat = np.load(self.txt_files[self.file_i])

                assert len(img_dat) == len(txt_dat)

                text_data.append(img_dat)
                image_data.append(txt_dat)
                n_loaded += 1
                assert len(text_data[-1]) == len(image_data[-1])
                assert len(text_data[0]) == len(image_data[0])
            except Exception as e:
                print("error loading files", self.img_files[self.file_i], self.txt_files[self.file_i], e)
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
        n_files = len(self.img_files)
        num_rows_per_file = 500000
        return n_files * num_rows_per_file // self.batch_size

    def __next__(self):
        if self.batch_i >= len(self.image_data):
            if self.file_i >= len(self.img_files):
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

        self.ds_train = IterableImageTextPairDataset(path_train, batch_size, )
        self.ds_test = IterableImageTextPairDataset(path_val, batch_size, )

    def train_dataloader(self):
        return DataLoader(self.ds_train, num_workers=4, batch_size=None, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.ds_test, num_workers=1, batch_size=None, shuffle=False)
