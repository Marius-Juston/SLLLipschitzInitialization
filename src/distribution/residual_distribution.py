import os.path

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from torch.nn import ReLU

from models.layers import safe_inv
from off_term_distributions import expected_value

matplotlib.use('TkAgg')
import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')


def w_variance_torch(nl, dl=None):
    if isinstance(nl, np.ndarray):
        nl = torch.from_numpy(nl)
    if isinstance(dl, np.ndarray):
        dl = torch.from_numpy(dl)

    if (dl is None):
        dl = nl

    return 1 / (nl + (dl - 1) * 1 / 2 * expected_value(nl))


def forward(x, W, b, act):
    T = torch.abs(torch.einsum('ik,kj -> ij', W, W.T)).sum(1)
    t = safe_inv(T)

    res = F.linear(x, W, b)

    act_res = act(res)

    num = W.T * res * 2

    res = t * act_res
    res = 2 * F.linear(res, W.T)
    out = x - res

    denom = t

    return out, x, res  # , num, denom


def covariance_value(Nmax=50, graph=False, std=1., dl=1, normal=True, save=False, use_saved=False, Nmin=2):
    device = 'cuda'

    save_file = "covariance_data.csv"

    Ns = list(range(1, Nmax + 1))

    dl = dl
    std = std

    act = ReLU()
    # act = lambda x: x

    means_f = []
    means_s = []
    means_c = []

    var_f = []
    var_s = []
    var_c = []

    names = [
        r'Mean $x_l$',
        r'Mean $2W_l^T T_l^{-1}\sigma\left(u_l\right)$',
        r'Mean $x_l 2W_l^T T_l^{-1}\sigma\left(u_l\right)$',
        r'Mean $E[x_l] E[2W_l^T T_l^{-1}\sigma\left(u_l\right)$]',
    ]

    loaded_saved = False

    if os.path.exists(save_file) and use_saved:
        data = np.genfromtxt(save_file, delimiter=',', skip_header=1)

        loaded_saved = not (data is None)

        if loaded_saved:
            nls = data[:, 0].astype(np.uint8)
            dls = data[:, 1].astype(np.uint8)
            means = data[:, 2:6].T
            vars = data[:, 6:9].T

            nls = torch.from_numpy(nls)
            dls = torch.from_numpy(dls)
            means = torch.from_numpy(means)
            vars = torch.from_numpy(vars)

    if not loaded_saved:
        for n in Ns:
            iterations = max(Nmin, math.floor(Nmax ** 2 / n ** 2))

            firsts = []
            seconds = []
            combined = []

            for i in range(iterations):
                x = torch.normal(0, 1, size=(n,), device=device)

                if not normal:
                    # x.uniform_(0, np.sqrt(3) * 2)
                    x.normal_(torch.e, torch.pi)
                W = torch.normal(0., std ** 2, size=(n * dl, n), device=device)
                b = torch.zeros(n * dl, device=device)

                _, first, second = forward(x, W, b, act)

                firsts.append(first)
                seconds.append(second)

                prod = first * second
                combined.append(prod)

            f = torch.hstack(firsts)
            s = torch.hstack(seconds)
            c = torch.hstack(combined)

            if graph:
                fig, axes = plt.subplots(ncols=3)

                for i, (n, z) in enumerate(zip(names, [f, s, c])):
                    count, bins = torch.histogram(z.cpu(), bins=100)
                    ax: Axes = axes[i]

                    ax.stairs(count, bins)
                    ax.set_title(n)

                plt.show()

            m_f = f.mean().item()
            m_s = s.mean().item()
            m_c = c.mean().item()

            means_f.append(m_f)
            means_s.append(m_s)
            means_c.append(m_c)

            var_f.append(f.var().item())
            var_s.append(s.var().item())
            var_c.append(c.var().item())

            print(n, m_f, m_s, m_c)

        means = [means_f, means_s, means_c]
        vars = [var_f, var_s, var_c]

        means = list(map(np.array, means))
        vars = list(map(np.array, vars))

        means.append(means[0] * means[1])

        nls = np.arange(1, means[0].shape[0] + 1)
        dls = nls * dl

        if save:
            all_data = np.stack([nls, dls, *means, *vars], axis=1)

            names = [
                'Nl',
                'Dl',
                'M x',
                'M S',
                'M xS',
                'M x * S',
                'V x',
                'V S',
                'V xS',
            ]

            delim = ','

            np.savetxt(save_file, all_data, delimiter=delim, header=delim.join(names))

    fig, axs = plt.subplots(nrows=2, ncols=len(means))
    for i, (n, m) in enumerate(zip(names, means)):
        print(i)
        Ns = np.arange(1, m.shape[0] + 1)

        ax: Axes = axs[0][i]
        ax.plot(Ns, m)
        ax.set_title(n)

        if i < len(vars):
            ax: Axes = axs[1][i]
            ax.plot(Ns, vars[i])

    expected_value_xtwxt = dls * w_variance_torch(dls, nls).cpu().numpy() * 1 / 2

    axs[0][2].plot(nls, expected_value_xtwxt, label='theoretical')

    N = 100
    sw = 1
    sb = 1

    if not normal:
        x_m = torch.e
        sx = torch.pi
    else:
        x_m = 0
        sx = 1

    w = lambda: torch.normal(0, sw, size=(N,))
    b = lambda: torch.normal(0, sb, size=(N,))

    x = lambda: torch.normal(x_m, sx, size=(N,))

    results = []

    for index, (nl, dl) in enumerate(zip(nls, dls)):
        print(index, nl.item(), dl.item())

        output = []

        for method in [simp1]:
            result = method(x, w, b, nl, dl)

            output.append(result.item())

        print(means[2][index].item(), output)

        results.append(result)

    axs[0][2].plot(nls, results, label='stepped')


def simp1(x, w, b, nl, dl):
    result = 0

    x1 = x()
    w1 = w()

    for _ in range(dl):
        n1 = x1 ** 2 * w1 ** 2

        n2 = x1 * w1 * b()

        n3 = 0

        for _ in range(nl - 1):
            n3 += (x1 * x() * w1 * w())

        d1 = w1 ** 2

        for _ in range(nl - 1):
            d1 += (w() ** 2)

        d2 = 0

        for _ in range(dl - 1):
            dd2 = 0.

            for _ in range(nl):
                dd2 += w1 * w()

            d2 += torch.abs(dd2)

        n = n1 + n2 + n3
        d = d1 + d2

        result += n / d

    result = result.mean()

    return result


def verify_formula():
    dl = 10
    nl = 2

    W = torch.arange(nl * dl).reshape((dl, nl)) + 10

    b = torch.zeros(dl)

    x = torch.arange(nl) + 100

    act = ReLU()

    W = W.to(torch.float)
    b = b.to(torch.float)
    x = x.to(torch.float)

    out, first, second = forward(x, W, b, act)

    print(out, first, second)


if __name__ == '__main__':
    # verify_formula()

    covariance_value(Nmax=100, Nmin=20_000, graph=False, dl=10, normal=False, save=False, use_saved=True)
    # covariance_value(Nmax=50, graph=False, dl=1, normal=False)

    # plt.legend()
    plt.show()
