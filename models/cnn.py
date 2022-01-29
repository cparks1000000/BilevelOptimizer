import torch.nn as nn


# Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_net = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3),
            nn.ReLU(),
        )
        # Convolution will reduce number of "pixels", in this case from 28 -> 22
        # Go from convolutional to classification vector
        self.classifier = nn.Sequential(
            nn.Linear(8*22*22, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        x = self.cnn_net(x)
        # Review x, so it is a vector of 8*22*22
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
