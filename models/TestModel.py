import torch.nn as nn
from models.HighLevelLayer import HighLevelLayer


class TestModel(nn.Module):
    """Self defined model for our neural network"""
    def __init__(self, input_size: int):
        super(TestModel, self).__init__()
        self.flatten = nn.Flatten()
        # Noah: Inductive bias needs to be added into our models
        self.net = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            HighLevelLayer(10, 16),
        )

    def forward(self, x):
        # x = x.view(-1)  # This one only works with marquardt.py
        x = self.flatten(x)  # This one only works with main.py because of batching!!!
        logits = self.net(x)
        return logits
