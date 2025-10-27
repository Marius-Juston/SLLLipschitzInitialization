from typing import Type

import numpy as np
import torch

from distribution import decay

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

torch.set_float32_matmul_precision('high')

from sklearn.datasets import fetch_covtype
from torch import nn
from torch.nn import Sequential, Softmax, ReLU
from torch.optim import Adam
from torch.utils.data import TensorDataset, random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F


def safe_inv(x):
    mask = x == 0
    x_inv = x ** (-1 / 2)
    x_inv[mask] = 0
    return x_inv


class SDPBasedLipschitzResidualLinearLayer(nn.Linear):
    INDEX = 0

    DECAYS = []

    CUM_DECAYS = []

    def compute_cummulative_decay(self):
        SDPBasedLipschitzResidualLinearLayer.CUM_DECAYS = 1 / np.cumprod(
            SDPBasedLipschitzResidualLinearLayer.DECAYS[::-1])[::-1]

    def __init__(self, cin, inner_dim, activation: Type[nn.Module] = nn.ReLU, better_initalization=False,
                 gradient=False):
        super(SDPBasedLipschitzResidualLinearLayer, self).__init__(in_features=cin, out_features=inner_dim, bias=True)

        self.class_index = SDPBasedLipschitzResidualLinearLayer.INDEX
        SDPBasedLipschitzResidualLinearLayer.INDEX += 1

        if activation.__name__ == 'VReLU':
            self.activation = activation(inner_dim)
        else:
            self.activation = activation()

        self.q = nn.Parameter(torch.randn(cin, dtype=torch.float))

        if better_initalization:
            with torch.no_grad():
                self.q.fill_(1.)
                self.bias.normal_(0., 1.0)
                self.weight.normal_(0, 1.0)

            if gradient:
                self.weight.register_hook(self.gradient_hook)
                self.bias.register_hook(self.gradient_hook)
        else:
            nn.init.xavier_normal_(self.weight, gain=torch.nn.init.calculate_gain('relu'))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.decay = decay(self.weight.shape[0], self.weight.shape[1])

        SDPBasedLipschitzResidualLinearLayer.DECAYS.append(self.decay)

        self.compute_cummulative_decay()

    def compute_t(self):
        q = torch.exp(self.q)
        q_inv = torch.exp(-self.q)
        t = torch.abs(torch.einsum('i,ik,kj,j -> ij', q_inv, self.weight.T, self.weight, q)).sum(1)
        t = safe_inv(t)
        return t

    def forward(self, x):
        t = torch.diag(self.compute_t())

        W = F.linear(self.weight, t)

        res = F.linear(x, W, self.bias)
        res = self.activation(res)
        return res

    def gradient_hook(self, grad):
        # print(self.class_index, SDPBasedLipschitzResidualLinearLayer.CUM_DECAYS[self.class_index], grad.var().item(), (grad * SDPBasedLipschitzResidualLinearLayer.CUM_DECAYS[self.class_index]).var().item())
        return grad * SDPBasedLipschitzResidualLinearLayer.CUM_DECAYS[self.class_index]


class SequentialClassifier(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_layer=10, better_initialization=False,
                 activation: Type[nn.Module] = nn.ReLU, gradient=False):
        super().__init__()

        start = 32
        max_size = 2048

        layers = []

        input_size = n_inputs
        output_size = start

        for i in range(n_layer // 2):
            print(input_size, output_size)
            layers.append(SDPBasedLipschitzResidualLinearLayer(input_size, output_size,
                                                               better_initalization=better_initialization,
                                                               activation=activation, gradient=gradient))

            input_size = output_size

            output_size = min(max_size, output_size * 2)

        for i in range(n_layer // 2, n_layer):
            print(input_size, output_size)
            layers.append(SDPBasedLipschitzResidualLinearLayer(input_size, output_size,
                                                               better_initalization=better_initialization,
                                                               activation=activation, gradient=gradient))

            input_size = output_size

            output_size = max(n_outputs, output_size // 2)

        print(input_size, output_size)
        layers.append(SDPBasedLipschitzResidualLinearLayer(input_size, n_outputs,
                                                           better_initalization=better_initialization,
                                                           activation=activation, gradient=gradient))

        layers.append(Softmax(dim=1))

        model = Sequential(*layers)

        print(model)

        self.model = model

    def forward(self, x):
        return self.model(x)


def train_loop(dataloader, model, loss_fn, optimizer, summary_writer: SummaryWriter, epoch, prof=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    train_loss = 0

    step = 0

    for batch, (X, y) in enumerate(dataloader):
        if prof is not None:
            prof.step()

            if step >= 1 + 1 + 3:
                break
            step += 1

        # Compute prediction and loss
        pred = model(X)

        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches

    summary_writer.add_scalar('loss/train', train_loss, global_step=epoch)


def test_loop(dataloader, model, loss_fn, summary_writer, epoch):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    summary_writer.add_scalar('loss/test', test_loss, global_step=epoch)
    summary_writer.add_scalar('loss/correct', correct, global_step=epoch)

    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def vizualize_weights(model: SequentialClassifier, summary_writer: SummaryWriter, epoch):
    for n, p in model.named_parameters():
        summary_writer.add_histogram(n, p, global_step=epoch)


if __name__ == '__main__':
    torch.random.manual_seed(0)
    np.random.seed(0)

    data = fetch_covtype(data_home='./data', shuffle=True, random_state=0)

    input_data = torch.tensor(data.data, dtype=torch.float)
    target = torch.tensor(data.target, dtype=torch.long) - 1

    if torch.cuda.is_available():
        input_data = input_data.to(device='cuda')
        target = target.to(device='cuda')
        device = 'cuda'

    dataset = TensorDataset(input_data, target)

    train, test = random_split(dataset, [0.8, 0.2])

    batch_size = 64
    epochs = 25
    lr = 1e-3

    train_dataloader = DataLoader(train, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test, shuffle=True, batch_size=batch_size)

    n_layers = 15
    n_inputs = len(data.feature_names)
    n_outputs = target.max() + 1
    activation = ReLU

    gradient = True

    for n_layers in [15]:

        models = [
            # SequentialClassifier(n_inputs=n_inputs, n_outputs=n_outputs, n_layer=n_layers).to(device=device),
            SequentialClassifier(n_inputs=n_inputs, n_outputs=n_outputs, n_layer=n_layers, better_initialization=True,
                                 activation=activation, gradient=gradient).to(device=device)
        ]

        model_names = [
            'He',
            'better_initialization']

        for name, model in zip(model_names, models):
            print("Compiling model")

            model = torch.compile(model, mode='reduce-overhead')

            print("Training model")
            optimizer = Adam(model.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()
            summary_writer = SummaryWriter(
                log_dir=f'runs/{name}_lr{lr}_n{n_layers}_act{activation.__name__}_g{int(gradient)}')
            # summary_writer.add_graph(model, next(iter(train_dataloader))[0])

            for t in range(epochs):
                print(f"Epoch {t + 1}\n-------------------------------")
                train_loop(train_dataloader, model, loss_fn, optimizer, summary_writer, t)

                vizualize_weights(model, summary_writer, t)

                test_loop(test_dataloader, model, loss_fn, summary_writer, t)

            print("Done!")

        del models
