import dataclasses
from src.model import ModelConfig, Decoder
from src.utils import CustomDataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter


@dataclasses.dataclass(kw_only=True)
class TrainerConfig:
    model_config: ModelConfig
    device: str = 'cuda'

    batch_size: int = 64
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    train_csv_path: str
    val_csv_path: str
    log_dir: str = "./runs/"
    cli_log_interval: int = 100

    eval_interval: int = 100
    eval_steps: int = 500

    def __post_init__(self):
        self.n_bins = self.model_config.n_bins

class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config

        self.device = config.device
        self.model = Decoder(config.model_config).to(self.device)
        self.batch_size = config.batch_size
        self.grad_clip = config.grad_clip

        self.eval_interval = config.eval_interval
        self.eval_steps = config.eval_steps

        self.train_loader = CustomDataLoader(
            file_path=config.train_csv_path,
            batch_size=config.batch_size,
            n_bins=config.n_bins
        )

        self.cli_log_interval = config.cli_log_interval

        self.writer = SummaryWriter(config.log_dir)

        self.val_loader = CustomDataLoader(
            file_path=config.val_csv_path,
            batch_size=config.batch_size,
            n_bins=config.n_bins
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            betas=(config.beta1,config.beta2),
            eps=config.eps
        )

    def _get_hparams(self):
        m_p =self.config.model_config
        return {
            "model_dim": m_p.model_dim,
            "n_layers": m_p.n_layers,
            "n_heads": m_p.n_heads,
            "bias": m_p.bias,
            "lr": self.config.lr,
            "beta1": self.config.beta1,
            "beta2": self.config.beta2,
            "eps": self.config.eps
        }

    def eval(self, steps: int):
        self.model.eval()
        self.val_loader._reset()
        total_correct = 0.0

        with torch.no_grad():
            for step in range(steps):
                X, y = next(self.val_loader)
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss = self.loss_fn(logits,y).item()
                correct = (logits.max(-1)[1] == y.max(1)[1]).sum().item()
                total_correct += correct

        acc = total_correct/(steps * self.batch_size)
        loss = loss/steps

        return acc, loss

    def _train_step(self, X, y):
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(X)
        loss = self.loss_fn(out, y)
        loss.backward()

        norm = clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss.item(), norm
    
    def train_hp(self, steps:int):
        for step in range(steps):
            X, y = next(self.train_loader)

            X = X.to(self.device)
            y = y.to(self.device)

            loss, _ = self._train_step(X, y)
            self.writer.add_scalar(f'loss/train', loss, step)


        acc, eval_loss = self.eval(len(self.val_loader))

        res = {
            "val_acc": acc,
            "val_loss": eval_loss
        }

        self.writer.add_hparams(self._get_hparams(), res)

        return res


    def train(self, steps: int):
        running_loss = 0.0
        print(f'Training for {steps} steps with a batch size of {self.batch_size}')

        for step in range(steps):
            X, y = next(self.train_loader)

            X = X.to(self.device)
            y = y.to(self.device)

            loss, norm = self._train_step(X, y)
            self.writer.add_scalar(f'loss/train', loss, step)
            running_loss += loss

            if step%self.cli_log_interval == self.cli_log_interval-1:
                print(f'Step {step:5d}: Loss: {running_loss/self.cli_log_interval:5f} Norm: {norm}')
                running_loss = 0.0

            if step%self.eval_interval == self.eval_interval-1:
                print(f'Calculating evaluation metrics for {self.eval_steps} steps')
                acc, eval_loss = self.eval(self.eval_steps)
                print(f'Val acc: {acc} Val loss: {eval_loss}')
                self.writer.add_scalar(f'loss/val', eval_loss, step)
                self.writer.add_scalar(f'acc/val', acc, step)