from collections.abc import Callable
from typing import Iterator, List, TypeVar, Tuple

import torch
from torch import Tensor, nn
import functools


# LM loops through each epoch (probably does not handle batches), gets the outputs from model,
# finds the gradient (usually loss.backward), updates parameters (usually optimizer.step)
# Just use Callable, not Callable[[Tensor], Tensor]
from models.TestModel import TestModel


def LM(toy_input: Tensor, toy_target: Tensor, model: nn.Module, loss: Callable, n_iter: int = 10) -> List[Tensor]:
    alpha: float = 1e-3
    loss_hist = []
    # J included this so that you could use loss function


    # Epoch loop
    for _ in range(n_iter):
        model.train()
        out: Tensor = model(toy_input)#.unsqueeze(1)
        loss_out: Tensor = loss(out, toy_target)
        print(loss_out.item())

        # Store the previous loss, so you can compare after the step
        previous_loss: Tensor = torch.clone(loss_out)

        change = calc_change(model, toy_input, toy_target, alpha)

        ###
        # I don't think this even does anything because the
        # grads are never stored in the model with .grad()
        # model.zero_grad()

        ### STEP
        model = take_step(model, change, forward=True)

        out = model(toy_input)#.unsqueeze(1)
        loss_out = loss(out, toy_target)

        if loss_out < previous_loss:
            print("Successful iteration")
            loss_hist.append(loss_out)
            alpha /= 10
        else:
            print("Augmenting step size")
            alpha *= 10
            model = take_step(model, change, forward=False)

        ###

    return loss_hist


def calc_change(model: nn.Module, input: Tensor, labels: Tensor, alpha: float):
    """Calculates change required to model parameters using LM method"""
    ### BACKWARD
    # .grad() returns the gradients whereas .backward() stores them in grad attribute
    jacobian = torch.tensor([])
    output = model(input)
    for i in range(len(output)):
        error = torch.abs(output-labels)[i]
        temp: Tuple[Tensor] = torch.autograd.grad(error, to_list(model.parameters()), create_graph=True)
        gradient_vector: Tensor = torch.tensor([])
        for g in temp:
            gradient_vector = torch.cat([gradient_vector, g.contiguous().view(-1)])
        torch.cat([jacobian, gradient_vector.unsqueeze(0)])


    model.eval()
    hessian: Tensor
    gradient_vector: Tensor
    # Can we use function.hessian here? (it is still in beta)
    hessian, gradient_vector = eval_hessian(gradients, model)

    # change is a Tensor[float] not float(?)
    change = -1 * (alpha * torch.eye(hessian.shape[-1]) + hessian).inverse().mm(
        gradient_vector).detach()

    return change


def take_step(model: nn.Module, change, forward=True):
    count = 0
    for p in model.parameters():
        mm = torch.Tensor([p.shape]).tolist()[0]
        # Reduce all values in mm by multiplying them, start with value 1
        num = int(functools.reduce(lambda x, y: x * y, mm, 1))
        p.requires_grad = False
        if forward:
            p += change[count:count + num, :].reshape(p.shape)
        else:
            p -= change[count:count + num, :].reshape(p.shape)
        count += num
        p.requires_grad = True
    return model


T = TypeVar('T')


def to_list(iterator: Iterator[T]) -> List[T]:
    output: List[T] = []
    element: T
    for element in iterator:
        output.append(element)
    return output


def approximate_hessian(loss_grad: Tuple[Tensor], model: nn.Module):
    pass


# Might change this to an approximation
def eval_hessian(loss_grad: Tuple[Tensor], model: nn.Module):
    gradient_vector: Tensor = torch.tensor([])
    for g in loss_grad:
        gradient_vector = torch.cat([gradient_vector, g.contiguous().view(-1)])
    length = len(gradient_vector)
    hessian: Tensor = torch.zeros(length, length)
    for i in range(length):
        second_derivatives: Tuple[Tensor] = torch.autograd.grad(gradient_vector[i], to_list(model.parameters()),
                                                                create_graph=True)
        g2: Tensor = torch.tensor([])
        for g in second_derivatives:
            g2 = torch.cat([g2, g.contiguous().view(-1)])
        hessian[i] = g2
    return hessian, gradient_vector.unsqueeze(1)


def main():
    h = 20
    w = 20
    b = 2
    # Use this one if you are using convolution layer
    # toy_input = torch.randn(1, 3, h, w)
    toy_input = torch.randn(b, h, w)
    toy_target: Tensor = torch.rand(b, 16)
    loss_fn = nn.MSELoss()
    # toy_model = NewModel(3, w, h)
    toy_model = TestModel(w*h)
    loss_history = LM(toy_input, toy_target, toy_model, loss_fn)

    print(list(map(lambda x: x.item(), loss_history)))


main()
