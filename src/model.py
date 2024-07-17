import torch 
import torch.nn as nn 

from utils import ModelConfig

#model v0
class BasicModel(nn.Model):
    def __init__(self, num_layers=32, bins=8):
        super().__init__()

        self.num_layers = num_layers
        self.bins = bins

        self.enc_layer = nn.TransformerEncoderLayer(
            d_model=75,
            nhead=5,
            batch_first=True
        )

        self.enc = nn.TransformerEncoder(
            self.enc_layer,
            self.num_layers
        )

        self.fc = nn.Linear(
            in_features=75,
            out_features=self.bins    
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.fc(x)
        return x