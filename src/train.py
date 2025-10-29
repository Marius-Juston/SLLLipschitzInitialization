import json
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import fetch_covtype
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

from distribution.train import train_loop, test_loop
from models.models import LipschitzLinearNetwork


def vizualize_weights(model: LipschitzLinearNetwork, summary_writer: SummaryWriter, epoch):
    for n, p in model.named_parameters():
        summary_writer.add_histogram(n, p, global_step=epoch)


class LinearNetwork(nn.Module):

    def __init__(self, in_features, out_features, n_dense=15, dense_inner_dim=256, device=None, dtype=None):
        super(LinearNetwork, self).__init__()

        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.n_dense = n_dense
        self.dense_inner_dim = dense_inner_dim
        self.out_features = out_features
        self.in_features = in_features

        self.model = []

        in_features = in_features

        for _ in range(self.n_dense):
            self.model.append(nn.Linear(in_features, dense_inner_dim, **self.factory_kwargs))
            self.model.append(nn.ReLU())

            in_features = dense_inner_dim

        self.model.append(nn.Linear(in_features, out_features, **self.factory_kwargs))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    data = fetch_covtype(data_home='../data', shuffle=True, random_state=0)

    input_data = torch.tensor(data.data, dtype=torch.float)
    target = torch.tensor(data.target, dtype=torch.long) - 1

    if torch.cuda.is_available():
        input_data = input_data.to(device='cuda')
        target = target.to(device='cuda')
        device = 'cuda'
    else:
        device = 'cpu'

    # torch.set_default_device(device)

    dataset = TensorDataset(input_data, target)

    generator = torch.Generator(device='cpu')

    train, test = random_split(dataset, [0.8, 0.2], generator=generator)

    batch_size = 64
    epochs = 25
    lr = 1e-3
    dense_inner_dim = 64

    lipschitz_network = False

    for n_layers in [5, 15, 30]:

        n_inputs = len(data.feature_names)
        n_outputs = target.max() + 1

        for bias_init in [True, ]:

            for sample in range(10):
                train_dataloader = DataLoader(train, shuffle=True, batch_size=batch_size)
                test_dataloader = DataLoader(test, shuffle=True, batch_size=batch_size)

                if lipschitz_network:
                    model = LipschitzLinearNetwork(in_features=n_inputs, out_features=n_outputs, n_dense=n_layers,
                                                   bias_init=bias_init, device=device, dense_inner_dim=dense_inner_dim)

                else:
                    model = LinearNetwork(in_features=n_inputs, out_features=n_outputs, n_dense=n_layers, device=device,
                                          dense_inner_dim=dense_inner_dim)

                name = model.__class__.__name__

                print("Compiling model")

                model = torch.compile(model, mode='max-autotune')

                print("Training model")
                optimizer = Adam(model.parameters(), lr=lr)
                loss_fn = nn.CrossEntropyLoss()
                summary_writer = SummaryWriter(
                    log_dir=f'../runs{'_lipschitz' if lipschitz_network else ''}/{name}_lr{lr}_n{n_layers}_b{bias_init}_{sample}')

                train_losses = []
                test_losses = []
                accuracies = []

                for t in range(epochs):
                    print(f"Epoch {t + 1}\n-------------------------------")
                    train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, summary_writer, t)

                    vizualize_weights(model, summary_writer, t)

                    test_loss, accuracy = test_loop(test_dataloader, model, loss_fn, summary_writer, t)

                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                    accuracies.append(accuracy)

                results = {
                    'name': name,
                    'n_layers': n_layers,
                    'bias_init': bias_init,
                    'sample': sample,
                    'dense_inner_dim': dense_inner_dim,
                    'lr': lr,
                    'batch_size': batch_size,
                    'epochs': epochs,
                }

                path = Path(summary_writer.get_logdir())

                with open(path / 'data.json', 'w') as f:
                    json.dump(results, f)

                print("Done!")
