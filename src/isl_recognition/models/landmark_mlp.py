"""MLP classifier for MediaPipe static-hand landmark features."""

import torch
from torch import nn


class LandmarkSignMLP(nn.Module):
    architecture_name = "static_landmark_mlp"

    def __init__(self, num_classes: int = 26, input_features: int = 134) -> None:
        super().__init__()
        self.input_features = input_features
        self.network = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
