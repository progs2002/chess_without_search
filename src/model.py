import dataclasses

import pandas as pd

from tokenizer import tokenize
import torch

@dataclasses.dataclass()
class ModelConfig:
    BATCH_SIZE: int

class CustomDataLoader:
    def __init__(self, file_path:str, batch_size:int=64):
        self.file_path = file_path
        self.batch_size = batch_size
        self.csv_iterator = pd.read_csv(file_path, iterator=True, dtype=str)

    def get_rows(self):
        df = self.csv_iterator.get_chunk(self.batch_size)
        return df.drop('score', axis=1), df['score']
    
    def get_batch(self):
        X, y = self.get_rows()
        X = X.apply(tokenize, axis=1)
        
        X = torch.stack(
            [t for t in X]
        )

        y = torch.from_numpy(
            y.to_numpy(dtype=int)
        )

        return X, y