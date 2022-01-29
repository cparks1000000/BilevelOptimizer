import torch


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # enumerate() is just used to get the times you have iterated placed into batch
    for batch, (features, labels) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(features)

        ###
        # LM will customize this
        loss = loss_fn(pred, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ###

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(features)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for features, labels in dataloader:
            pred = model(features)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")