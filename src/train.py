import os
import dataclasses
from src.model import ModelConfig, Decoder
from src.utils import CustomDataLoader

# from typing import Self, Tuple

import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter


@dataclasses.dataclass(kw_only=True)
class TrainerConfig:
    train_csv_path: str
    val_csv_path: str

    checkpoint_dir: str
    log_dir: str = "./runs/"

    device: str = 'cuda'
    
    n_bins: int = 32

    batch_size: int = 64
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    log_lr: bool = False

    warmup_iters: int = 1000
    lr_decay_iters: int = 7000
    min_lr: int = 3e-5

    cli_log_interval: int = 100

    loss_fn_weight: torch.Tensor|None = None

    eval_interval: int = 400
    eval_steps: int = 1200

    checkpoint_interval: int = 10000

    def __post_init__(self):
        if self.loss_fn_weight is not None:
            self.weighted_loss = True
            self.loss_fn_weight = self.loss_fn_weight.to(self.device)
        else:
            self.weighted_loss = False

class Trainer:
    def __init__(self, config: TrainerConfig, model:nn.Module):
        self.config = config

        self.model = model.to(config.device)

        self.device = config.device
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

        self.weighted_loss = config.weighted_loss
        self.loss_fn = nn.CrossEntropyLoss(
            weight=config.loss_fn_weight
        )

        self.lr = config.lr
        self.log_lr = config.log_lr
        self.warmup_iters = config.warmup_iters
        self.lr_decay_iters = config.lr_decay_iters
        self.min_lr = config.min_lr

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            betas=(config.beta1,config.beta2),
            eps=config.eps
        )

        self.checkpoint_dir = config.checkpoint_dir
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.checkpoint_interval = config.checkpoint_interval

        self.scheduler = ReduceLROnPlateau(self.optimizer)
        self.global_step_offset = 0

    @classmethod
    def init_from_checkpoint(cls, checkpoint_path: str): #-> Tuple[Self, torch.nn.Module]:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model_config = checkpoint["model_config"]
        model_dict = checkpoint["model_dict"]
        print(model_dict)
        trainer_config = checkpoint["trainer_config"]
        optimizer_dict = checkpoint["optimizer_dict"]
        step = checkpoint["step"]

        model = Decoder(model_config)
        model.load_state_dict(model_dict)

        trainer = Trainer(trainer_config, model)
        trainer.optimizer.load_state_dict(optimizer_dict)
        trainer.global_step_offset = step

        return trainer, model


    def _get_hparams(self):
        model_hparams = self.model._get_hparams()
        trainer_hparams = {
            "batch_size": self.batch_size,
            "grad_clip": self.grad_clip,
            "lr": self.config.lr,
            "beta1": self.config.beta1,
            "beta2": self.config.beta2,
            "eps": self.config.eps,
            "warmup_iters": self.warmup_iters,
            "lr_decay_iters": self.lr_decay_iters,
            "min_lr": self.min_lr,
            "weighted_loss": self.weighted_loss
        }

        return model_hparams | trainer_hparams

    #borrowed from karpathy's nano gpt :)
    def _get_lr(self, it):
        if it < self.warmup_iters:
            return self.lr * it / self.warmup_iters
        if it > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (it -self. warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
        return self.min_lr + coeff * (self.lr - self.min_lr)

    def _update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def save_checkpoint(self, step: int, checkpoint_path: str):
        checkpoint = {
            "model_config": self.model.config,
            "trainer_config": self.config,
            "model_dict": self.model.state_dict(),
            "optimizer_dict": self.optimizer.state_dict(),
            "step": step
        }

        torch.save(checkpoint, checkpoint_path)

    def restore_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model_config = checkpoint["model_config"]
        model_dict = checkpoint["model_dict"]
        trainer_config = checkpoint["trainer_config"]
        optimizer_dict = checkpoint["optimizer_dict"]
        step = checkpoint["step"]

        assert trainer_config == self.config
        assert model_config == self.model.config

        self.model.load_state_dict(model_dict)
        self.optimizer.load_state_dict(optimizer_dict)
        self.global_step_offset = step

    def eval(self, steps: int):
        self.model.eval()
        self.val_loader._reset()
        total_correct = 0.0
        running_loss = 0.0

        with torch.no_grad():
            for step in range(steps):
                X, y = next(self.val_loader)
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)[:,-1,:]
                loss = self.loss_fn(logits, y).item()
                running_loss += loss

                correct = (logits.max(-1)[1] == y).sum().item()
                total_correct += correct

        acc = total_correct/(steps * self.batch_size)
        loss = running_loss/steps

        return acc, loss

    def _train_step(self, X: torch.Tensor, y: torch.Tensor):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        logits = self.model(X)[:,-1,:]
        loss = self.loss_fn(logits, y)
        loss.backward()

        norm = clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss.item(), norm

    def train(self, steps: int):
        running_loss = 0.0
        print(f'Training for {steps} steps with a batch size of {self.batch_size}')

        self.writer.add_text('run params', str(self._get_hparams()))

        for step_count in range(steps):
            step = self.global_step_offset + step_count
            X, y = next(self.train_loader)

            X = X.to(self.device)
            y = y.to(self.device)

            new_lr = self._get_lr(step)
            self._update_lr(new_lr)

            if self.log_lr:
                self.writer.add_scalar('lr', new_lr, step)

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

            if step%self.checkpoint_interval == self.checkpoint_interval-1:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}steps.pt")
                print(f'Saving checkpoint at {step} steps')
                self.save_checkpoint(step, checkpoint_path)