import re
from abc import ABCMeta, abstractmethod
from os.path import isfile

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class PennTreeBankDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, win_len, target_len, root = "../data/penn-treebank/",
                 is_train: bool = False, is_valid: bool = False, is_test: bool = False,
                 include_uppercase=False) -> None:
        super().__init__()

        if (is_train + is_valid + is_test) != 1:
            raise ValueError("Only one of is_train, is_val, and is_test can be True")

        self.root = root
        self.target_len = target_len
        self.win_len = win_len

        self.is_train = is_train
        self.is_valid = is_valid
        self.is_test = is_test

        self.include_uppercase = include_uppercase

        self._setup_alphabet()

    def _setup_alphabet(self):
        if not self.include_uppercase:
            self.CHAR_LIST = r'abcdefghijklmnopqrstuvxyz0123456789 .!?:,\'%-()/$|&;[]"'
        else:
            self.CHAR_LIST = r'abcdefghijklmnopqrstuvxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?:,\'%-()/$|&;[]"'

        chars = [c for c in self.CHAR_LIST]
        chars.insert(0, "UNK")
        self.char2index = dict((c, i) for i, c in enumerate(chars))
        self.index2char = dict((i, c) for i, c in enumerate(chars))

    @property
    def filename(self):
        if self.is_train:
            filename = "train"
        elif self.is_valid:
            filename = "valid"
        elif self.is_test:
            filename = "test"

        return filename

    @property
    def num_char(self):
        return len(self.CHAR_LIST)

    def _convert_char2index(self, raw_data):
        text_data = []
        for c in raw_data:
            idx = 0
            try:
                idx = self.char2index[c]
            except:
                pass  # UNK will be inserted

            text_data.append(idx)

        return torch.LongTensor(text_data)

    @abstractmethod
    def _convert_raw(self):
        pass

    @abstractmethod
    def _setup_data(self):
        pass

class PennTreeCharDataset(PennTreeBankDataset):

    def __init__(self, win_len, target_len, root="../data/penn-treebank",
                 is_train: bool = False, is_valid: bool = False, is_test: bool = False,
                 include_uppercase: bool = False) -> None:
        super().__init__(win_len, target_len, root, is_train, is_valid, is_test, include_uppercase)

        self._setup_data()

    def _setup_data(self):

        hdf5_filename = f"{self.root}/ptb.{self.filename}.hdf5"

        if isfile(hdf5_filename):
            with h5py.File(hdf5_filename, "r") as f:
                self.data = f["data"][()]
        else:
            self.data = self._convert_raw()
            with h5py.File(hdf5_filename, "w") as f:
                f.create_dataset("data", data = self.data)

    def _convert_raw(self):
        raw_filename = f"{self.root}/ptb.{self.filename}.txt"

        with open(raw_filename) as f:
            raw_data = f.read()

            """Filter"""
            if not self.include_uppercase:
                raw_data = raw_data.lower()

            raw_data = re.sub(r'<unk>', " ", raw_data)
            raw_data = re.sub(r"\n", " ", raw_data)
            raw_data = re.sub(r"[ ]+", " ", raw_data)

        return self._convert_char2index(raw_data)

    def __getitem__(self, idx: int):
        return [
            self.data[idx: idx + self.win_len],
            self.data[idx + self.win_len: idx + self.win_len + self.target_len]
        ]

    def __len__(self) -> int:
        return self.data.shape[0] - self.win_len - self.target_len





print("yes")