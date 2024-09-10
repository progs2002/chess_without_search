import dataclasses

import numpy as np
import pandas as pd

from src.tokenizer import tokenize_from_series

import torch
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cp_to_win_percent(cp):
    win = 0.5 + 0.5 * (2 / (1 + np.exp(-0.00368208 * cp)) - 1)
    return win

class CustomDataLoader:
    def __init__(self, file_path:str, batch_size:int=64, n_bins=32):
        self.file_path = file_path
        self.batch_size = batch_size
        self.n_bins = n_bins

        temp = np.linspace(0,1,n_bins)
        self.edges = (temp[1:] + temp[:-1]) / 2
        
        data_file = open(file_path)
        rows = data_file.readlines()
        self.len = len(rows) - 1
        data_file.close()

        self.csv_iterator = pd.read_csv(file_path, iterator=True, dtype=str)

    def _get_rows(self):
        try:
            df = self.csv_iterator.get_chunk(self.batch_size)
            return df.drop('score', axis=1), df['score']
        except StopIteration:
            self._reset()
            return self._get_rows()

    def __len__(self):
        return self.len // self.batch_size

    def _reset(self):
        self.csv_iterator = pd.read_csv(self.file_path, iterator=True, dtype=str)

    def __iter__(self):
        return self
    
    def _transform_features(self, x):
        tensors = x.apply(tokenize_from_series, axis=1)
        return torch.stack(
            [t for t in tensors]
        )

    def _transform_labels(self, x):
        x = x.astype('float') 
        bins = np.searchsorted(self.edges, x, side='left')

        return torch.from_numpy(bins)

    def __next__(self):
        X, y = self._get_rows()
        Xb = self._transform_features(X)
        yb = self._transform_labels(y)
        return Xb, yb