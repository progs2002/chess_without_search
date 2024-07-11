import dataclasses

import numpy as np
import pandas as pd

from tokenizer import tokenize_from_series
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@dataclasses.dataclass()
class ModelConfig:
    BATCH_SIZE: int

def cp_to_win_percent(cp):
    return 50 + 50 * (2 / (1 + np.exp(-0.00368208 * cp)) - 1)

class CustomDataLoader:
    def __init__(self, file_path:str, batch_size:int=64):
        self.file_path = file_path
        self.batch_size = batch_size
        
        data_file = open(file_path)
        rows = data_file.readlines()
        self.len = len(rows) - 1
        data_file.close()

        self.csv_iterator = pd.read_csv(file_path, iterator=True, dtype=str)

    def _get_rows(self):
        df = self.csv_iterator.get_chunk(self.batch_size)
        return df.drop('score', axis=1), df['score']

    def __len__(self):
        return self.len // self.batch_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        X, y = self._get_rows()

        if len(X) < self.batch_size:
            raise StopIteration
        else:
            tensors = X.apply(tokenize_from_series, axis=1)
            
            Xb = torch.stack(
                [t for t in tensors]
            )

            yb = torch.from_numpy(
                y.to_numpy(dtype=int)
            )

        return Xb, yb