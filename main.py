# import functools
# from collections.abc import Callable
# from typing import Iterator, List, TypeVar, Tuple
from torch import nn  # Create Neural Networks
from torch.utils.data import DataLoader, Dataset  # Load data
import torch.optim as optim  # Import a default optimizer
from torchvision import datasets  # Standard datasets to test things with
from torchvision.transforms import ToTensor, PILToTensor  # Convert PIL image or ndarray to tensor

from models.TestModel import TestModel
from loops import train_loop, test_loop


def main():
    # Hyper-parameters
    num_epochs: int = 5
    learning_rate: float = .001
    batch_size: int = 10
    alpha: float = .001

    # TODO: These datasets are not working. They are filled with 0s
    # Download a std dataset from torchvision
    # train_dataset = datasets.FashionMNIST(
    #     root="data",
    #     train=True,
    #     download=True,
    #     transform=ToTensor()
    #     # target_transform=None # Could change this to a one-hot encoding to improve performance
    # )
    # train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    #
    # test_dataset = datasets.FashionMNIST(
    #     root="data",
    #     train=False,
    #     download=True,
    #     transform=ToTensor()
    # )
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    # This dataset suffers from the same problem as the other one
    train_dataset = datasets.EMNIST(
        root="data",
        split="digits",
        download=True,
        train=True,
        transform=ToTensor()
    )
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    test_dataset = datasets.EMNIST(
        root="data",
        split="digits",
        download=True,
        train=False,
        transform=ToTensor()
    )
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
    print("Data done")

    # Our model instance with num_inputs passed in
    model = TestModel(28 * 28)

    # Basic loss function
    # Can't use MSE here
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.MSELoss()

    # Basic optimizer with no momentum
    optimizer = optim.SGD(model.parameters(), learning_rate)

    # TRAINING/TESTING LOOP
    # New epoch after you've gone through whole training set
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done")


if __name__ == '__main__':
    main()
