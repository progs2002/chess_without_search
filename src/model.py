import dataclasses

import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

from utils import CustomDataLoader

@dataclasses.dataclass()
class ModelConfig:
    n_layers: int = 32
    n_bins: int = 128

#model v0
class BasicModel(nn.Model):
    def __init__(self, config:ModelConfig):
        super().__init__()

        self.n_layers = config.n_layers
        self.n_bins = config.n_bins

        self.enc_layer = nn.TransformerEncoderLayer(
            d_model=75,
            nhead=5,
            batch_first=True
        )

        self.enc = nn.TransformerEncoder(
            self.enc_layer,
            self.n_layers
        )

        self.fc = nn.Linear(
            in_features=75,
            out_features=self.n_bins    
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.fc(x)
        return x

@dataclasses.dataclass(hash=False)
class TrainerConfig:
    model_config: ModelConfig

    train_csv_path: str
    val_csv_path: str

    batch_size: int = 64
    cli_log_steps: int = 1000
    log_dir: str = "./runs/"

    lr: float

class Trainer:
    def __init__(self, config: TrainerConfig):
        self.model_config = config.model_config
        self.model = BasicModel(self.model_config)

        self.batch_size = config.batch_size
        self.n_bins = config.model_config.n_bins
        self.train_csv_path = config.train_csv_path
        self.train_loader = CustomDataLoader(
            file_path=self.train_csv_path,
            batch_size=self.batch_size,
            n_bins=self.n_bins
        )

        self.cli_log_steps = config.cli_log_steps

        self.log_dir = config.log_dir
        self.writer = SummaryWriter(self.log_dir)

        self.val_csv_path = config.val_csv_path
        self.val_loader = CustomDataLoader(
            file_path=self.val_csv_path
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.lr = config.lr
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

    def train_step(self):

        running_loss = 0.0

        for Xb, yb in self.train_loader:
            self.optim.zero_grad()
            out = self.model(Xb)

            loss = self.loss_fn(out, yb)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()