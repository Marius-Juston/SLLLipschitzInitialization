from functools import partial
from typing import Type, Optional

import matplotlib
from torch import nn, Tensor, autograd
from torch.nn import ReLU, Softplus, GELU, Tanh, ELU, Sequential
from torch.nn.init import calculate_gain

from off_term_distributions import expected_value

matplotlib.use('TkAgg')
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_float32_matmul_precision('high')
import torch.nn.functional as F


def histogram(xs, bins=100, density=False):
    # Like torch.histogram, but works with cuda
    min, max = xs.min(), xs.max()
    counts = torch.histc(xs, bins, min=min, max=max)
    boundaries = torch.linspace(min, max, bins + 1)

    if density:
        spacing = (max - min) / bins

        N = xs.shape.numel()

        counts = counts / N / spacing

    return counts, boundaries


class Distribution:
    def __init__(self, N=10):
        # self.weight = torch.randn((N, N))
        self.weight = torch.arange(N ** 2).reshape((N, N)).to(dtype=torch.float)
        self.q = torch.ones(N)

    def compute_t(self):
        q = self.q
        q_inv = torch.reciprocal(self.q)
        t = torch.abs(torch.einsum('i,ik,kj,j -> ij', q_inv, self.weight, self.weight.T, q)).sum(1)
        return t


def w_variance(Ns, dl=None):
    if (dl is None):
        dl = Ns

    return 1 / (dl + (Ns - 1) * 1 / 2 * expected_value(dl).numpy())


def w_variance_torch(Ns, dl=None):
    if (dl is None):
        dl = Ns

    return 1 / (Ns + (dl - 1) * 1 / 2 * expected_value(Ns))


def check_validity(W, T):
    T = torch.diag(T)

    outputs = torch.linalg.eigvals(W.T @ W - T)

    negs = outputs.real <= 0

    print(torch.round(outputs.real, decimals=2))

    return torch.all(negs)


def check():
    N = 10

    dist = Distribution(N)

    W = dist.weight

    s = torch.abs(W.T @ W).sum(1)
    print(check_validity(W, s))

    T = dist.compute_t()
    print(check_validity(W, T))


def distribution(W):
    # n = W.shape[1]

    # q = torch.ones(n, device=W.device)
    # q_inv = torch.reciprocal(q)
    # T = torch.abs(torch.einsum('i,ik,kj,j -> ij', q_inv, W.T, W, q)).sum(1)

    T = torch.abs(torch.einsum('ik,kj -> ij', W.T, W)).sum(1)

    T = torch.reciprocal(torch.sqrt(T))

    return W @ torch.diag(T)


def distribution_test(Nmax=300, normalized=False, graph=False, set_n=False):
    Ns = np.arange(1, Nmax + 1)
    stds = []

    dl = 10

    Ds = np.maximum(dl * Ns, 1).astype(Ns.dtype)

    n_scale = 2

    for n, dl_n2 in zip(Ns, Ds):

        iterations = max(2, math.floor(Nmax ** 2 / n ** 2))

        transformed = []
        normal = []

        if set_n:
            n = n * n_scale

        for i in range(iterations):

            if not normalized:
                # W = torch.randn((n, n), device='cuda') # N(0, 1)
                # W = torch.empty((n, n), device='cuda')
                # W.bernoulli_(0.5)
                # W = (W * 2 - 1)
                W = torch.normal(0, 1, size=(dl_n2, n), device='cuda')  # N(0, n^2)

                # Uniform distribution
                # W = torch.rand(size=(dl_n2, n), device='cuda')  # N(0, n^2)
                # W = W * 2 - 1
                # W = W * np.sqrt(3)

            else:
                # W = torch.normal(0, std=1 / np.sqrt(c0 + c1 * n + c2 * n ** 2), size=(n, n), device='cuda')
                W = torch.normal(0, std=1, size=(dl_n2, n), device='cuda')
                # W = torch.normal(0, std=1/np.sqrt(c0 + c1 * n + c2 * n ** 2), size=(n, n), device='cuda')

            Wtrans = distribution(W)

            # if normalized:
            #     n = min(W.shape)
            #
            #     q = torch.ones(n, device=W.device)
            #     q_inv = torch.reciprocal(q)
            #     T = torch.diag(torch.einsum('i,ik,kj,j -> ij', q_inv, W.T, W, q))
            #
            #     T = torch.reciprocal(torch.sqrt(T))
            #
            #     Wtrans = W @ torch.diag(T)

            transformed.append(Wtrans)
            normal.append(W)

        t = torch.concat(transformed)

        symmetry = (t > 0).sum()
        print(symmetry, t.shape.numel(), symmetry / t.shape.numel())

        if graph:
            fig, (ax1, ax2) = plt.subplots(1, 2)

            ax1.hist(torch.stack(normal).flatten().cpu(), bins=100)
            ax2.hist(torch.stack(transformed).flatten().cpu(), bins=100)

            plt.show()

        stds.append(t.var().item())
        print(t.mean().item())

    data = np.array([Ns, stds]).T

    Ns = np.array(Ns)

    if set_n:
        Ns *= n_scale

    stds = np.array(stds)

    plt.plot(Ns, stds, label='normalized' if normalized else 'normal')

    print(np.max(stds * Ns / 2))

    if not normalized:
        pass
        # np.savetxt('std_data.csv', data, delimiter=',')
        plt.plot(Ns, w_variance(Ns, Ds), label='bound')
        plt.plot(Ns, 1 / (Ds), label='upper-bound')
        # plt.plot(Ns, 1/( Ns), label='upper-bound Nl')

    plt.grid()
    plt.yscale('log')
    plt.ylabel("Variance")
    plt.xlabel(f"W dimension size (${dl} n_l \\times n_l$)")

    plt.legend()
    plt.tight_layout()
    plt.savefig('../../figs/VariancePerDimensionSize.png', dpi=300)

    return 0


def uniform_test():
    n = 100000

    W = torch.normal(0, 1, (n, 1))
    b = torch.normal(0, 1, (n, 1))

    x = torch.empty((n, 1)) * 100

    # x.normal_(0, 1)
    # x.bernoulli_(0.5)
    x.uniform_(-10, 10)

    out = W * x + b

    total = (out > 0).sum().item()
    print(out.mean(), total, n, total / n)


def multi_layer_distribution_test(N=1000, n=10, L=5, multi: Optional[bool] = None):
    device = 'cuda'

    N = N

    n = n

    y1 = torch.normal(0, 1, size=(N, n), device=device)

    xl = y1

    # act = VReLU(n=n)
    act = ReLU()
    # act = Tanh()

    yls = [y1]

    if multi is None:
        dl_mul = 1
    elif multi:
        dl_mul = 2
    else:
        dl_mul = 0.5

    dls = []
    ns = []

    for i in range(L):
        nl = xl.shape[1]
        print(nl)

        dl = nl * dl_mul

        dl = max(1, min(int(dl), 4096 * 2))

        ns.append(nl)
        dls.append(dl)

        W = torch.randn((dl, nl), device=device)
        b = torch.randn(dl, device=device)
        # b = torch.zeros(dl, device=device)

        # nn.init.xavier_normal_(W)
        # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(W)
        # bound = 1 / np.sqrt(fan_in)
        # nn.init.uniform_(b, -bound, bound)  # bias init

        # b = torch.randn(n)

        T = torch.abs(torch.einsum('ik,kj -> ij', W.T, W)).sum(1)
        T = torch.reciprocal(torch.sqrt(T))
        T = torch.diag(T)

        yl = F.linear(xl, (W @ T), b)

        yls.append(yl)

        xl = act(yl)

        print((xl == 0).sum() / xl.shape.numel())

    vars = []

    for i, y in enumerate(yls):
        var = y.std() ** 2

        vars.append(var)

        print(vars)

        counts, bins = histogram(y, bins=100)

        # if i == 999:
        plt.stairs(counts.cpu(), bins.cpu())
        # plt.show()

    vars = torch.stack(vars).cpu()

    print(vars)

    plt.xlabel("Activation output $y_l$")
    plt.ylabel("Histogram Frequency")
    plt.tight_layout()
    plt.savefig("../../figs/HistVarianceActivation.png", dpi=300)
    plt.show()

    Ls = np.arange(1, L + 2)

    plt.plot(Ls[1:], vars[1:], label='sampled')

    Ls_v = np.arange(2, L + 2)

    dls = np.array(dls)
    ns = np.array(ns)

    print(ns, dls)

    decay_rate = decay(ns, dls)

    print(np.cumprod(decay_rate))

    plt.plot(Ls_v, np.cumprod(decay_rate), label='theoretical')
    plt.plot(Ls_v, np.cumprod(1 / 2 * ns / dls), label='theoretical upper')
    plt.title("Activation Variance within layers")
    # plt.ylim(bottom=.9, top=1.1)

    plt.xlabel("Layer number")
    plt.ylabel("Variance")
    plt.yscale('log')

    plt.grid()

    plt.legend()
    plt.tight_layout()

    plt.savefig("../../figs/VarianceActivation.png", dpi=300)

    plt.show()


class LinearL(nn.Module):
    INPUTS = []

    def __init__(self, n, activation: Type[nn.Module] = nn.ReLU, bias=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = activation()
        W = torch.randn((n, n))

        if bias:
            b = torch.randn(n)
        else:
            b = torch.zeros(n)

        # nn.init.xavier_normal_(W, gain=calculate_gain('relu'))
        # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(W)
        # bound = 1 / np.sqrt(fan_in)
        # nn.init.uniform_(b, -bound, bound)  # bias init

        # b.normal_(0, 10000)

        self.W = torch.nn.Parameter(W)
        self.b = torch.nn.Parameter(b)

    def forward(self, x: Tensor):
        T = torch.abs(torch.einsum('ik,kj -> ij', self.W.T, self.W)).sum(1)
        T = torch.reciprocal(torch.sqrt(T))
        T = torch.diag(T)

        self.yl = F.linear(x, self.W @ T, self.b)

        x_L1 = self.activation(self.yl)

        LinearL.INPUTS.append(x_L1)

        return x_L1


def gradient_multi_layer_distribution_test(N=10000, n=10, L=5, Nb=25, bias=True):
    device = 'cuda'

    N = N

    n = n

    w_grads = [[] for _ in range(L - 1)]
    # act = partial(VReLU, n=n)
    act = ReLU

    model = Sequential(*([LinearL(n, act, bias=bias)] * L)).to(device=device)

    # model = torch.compile(model, mode='reduce-overhead')

    y1 = torch.empty(size=(N, n), device=device)

    scaling_factor = 1

    for _ in range(Nb):
        LinearL.INPUTS.clear()
        print(_)

        y1.normal_(0, 1)

        xL_1 = model(y1)

        model.zero_grad()
        loss = xL_1.sum() * scaling_factor

        input_losses = autograd.grad(loss, LinearL.INPUTS[:len(w_grads)])

        for i, y in enumerate(input_losses):
            w_grads[i].append(y)

    vars = []
    decay_rate = decay(n, n)

    for i, var in enumerate(w_grads):
        i += 2

        print(i)

        W_grad = torch.stack(var)  # * np.sqrt(1 / (decay_rate ** (L - 1 - i + 1 + 1)))

        vars.append(W_grad.var().item())

        # counts, bins = torch.histogram(W_grad.cpu(), bins=100)
        counts, bins = histogram(W_grad, bins=100)

        # if i == 999:
        plt.stairs(counts.cpu(), bins.cpu())
    del w_grads

    plt.xlabel("Gradient Activation output $x_l$")
    plt.ylabel("Histogram Frequency")
    plt.tight_layout()
    plt.savefig(f"../../figs/HistGradVarianceActivation{'B' if bias else ''}.png", dpi=300)

    plt.show()

    vars = np.array(vars)

    print(vars)

    Ls = np.array(range(2, len(vars) + 2))

    plt.plot(Ls, vars, label=f'sampled {"($b_l = 0$)" if not bias else ""}')

    plt.plot(Ls, (decay_rate ** (L - 1 - Ls + 1 + 1)) * scaling_factor, label='theoretical')
    plt.plot(Ls, ((1 / 2) ** (L - 1 - Ls + 1 + 1)) * scaling_factor, label='theoretical upper')

    plt.title("Gradient Activation Variance within layers")

    plt.xlabel("Layer num")
    plt.ylabel("Variance")
    plt.yscale('log')

    plt.grid()

    plt.legend()
    plt.tight_layout()

    plt.savefig(f"../../figs/GradientVarianceActivation{'B' if bias else ''}.png", dpi=300)

    plt.show()


def decay(n, dl=None):
    return n / 2 * (w_variance(n, dl))


class VReLU(nn.Module):
    def __init__(self, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scaling = 1 / (w_variance_torch(n) * n)

        factor = torch.sqrt(2 * scaling)

        print(factor, w_variance_torch(n))

        self.register_buffer("factor", factor)

        self.relu = ReLU()

    def forward(self, x):
        return self.relu(self.factor * x)


if __name__ == '__main__':
    # multi_layer_distribution_test(L=12, n=4096 * 2, multi=False)
    # multi_layer_distribution_test(N=10_000, L=13, n=1, multi=True)
    # multi_layer_distribution_test(N=10_000, L=13, n=4096 * 2, multi=None)
    # gradient_multi_layer_distribution_test(L=15, n=2024, bias=False)
    # gradient_multi_layer_distribution_test(L=15, n=2024, bias=True)
    # uniform_test()

    # check()

    distribution_test(normalized=False, graph=False, set_n=False)
    # distribution_test(normalized=False, graph=False)

    plt.legend()
    plt.tight_layout()
    plt.show()
