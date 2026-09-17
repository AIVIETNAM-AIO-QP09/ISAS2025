"""Models preserve the original notebook's architecture."""

import torch
from torch import nn


class PoseEncoder(nn.Module):
    def __init__(self, input_dim=34, model_dim=64, emb_dim=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=4, batch_first=True),
            num_layers=3,
        )
        self.bottleneck = nn.Linear(model_dim, emb_dim)

    def forward(self, x):
        x = self.encoder(self.input_proj(x))
        return self.bottleneck(x).mean(dim=1)


class MILTransformer(nn.Module):
    def __init__(self, num_classes, input_dim=128, model_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers,
        )
        self.attention_weights = nn.Linear(model_dim, 1)
        self.classifier = nn.Sequential(nn.Linear(model_dim, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.transformer(x)
        weights = torch.softmax(self.attention_weights(x), dim=1)
        return self.classifier(torch.sum(weights * x, dim=1))
