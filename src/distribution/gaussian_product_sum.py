import matplotlib.pyplot as plt
import numpy as np
import scipy
import sympy
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.special import kv
from sympy import gamma

from distribution import histogram


def difference(v, n, s, abs=False):
    denom = sympy.gamma(n / 2) * (2 ** ((n - 1) / 2)) * np.sqrt(torch.pi) * s ** (n + 1)
    num = torch.abs(v) ** ((n - 1) / 2) * (1 + abs)

    bessel = scipy.special.kv((n - 1) / 2, torch.abs(v) / s ** 2)

    return num * bessel / float(denom)


def check_inner_sum(abs=False):
    N = 1_000_000_000

    n = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    s = 1

    total = torch.zeros((N, 1), device=device)

    for i in range(n):
        w1 = torch.normal(0, s, size=(N, 1), device=device)
        w2 = torch.normal(0, s, size=(N, 1), device=device)

        total += w1 * w2

        temp_total = total

        if abs:
            temp_total = total.abs()

        counts, steps = histogram(temp_total, bins=100_000, density=True)

        plt.stairs(counts.cpu(), steps.cpu())

        # theoretical_output = sum_function(steps, s)
        # plt.plot(steps, theoretical_output)

        theoretical_c = difference(steps, i + 1, s, abs=abs)

        plt.plot(steps, theoretical_c)

        plt.xlim(-3 * (1 - abs) + -.01 * abs, 3)

        plt.show()


def check_outer_sum():
    N = 500_000_000

    n = 10
    nl_ = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    s = 1

    total_outer = torch.zeros((N, 1), device=device)
    total_outer_abs = torch.zeros((N, 1), device=device)

    for nl in range(nl_):

        total = torch.zeros((N, 1), device=device)

        for dl in range(n):
            w1 = torch.normal(0, s, size=(N, 1), device=device)
            w2 = torch.normal(0, s, size=(N, 1), device=device)

            total += w1 * w2

        total_outer += total
        total_outer_abs += torch.abs(total)

        # counts, steps = histogram(total_outer, bins=100_000, density=True)
        # plt.stairs(counts.cpu(), steps.cpu(), label='gamma')

        counts, steps = histogram(total_outer_abs, bins=100_000, density=True)
        plt.stairs(counts.cpu(), steps.cpu(), label='actual')

        mean = total_outer_abs.mean()
        var = total_outer_abs.var()

        theta = var / mean
        k = mean / theta
        print(mean, var, theta, k)

        # distribution = torch.distributions.gamma.Gamma(k, 1 / theta)
        # total_outer_abs_sampled = distribution.sample((N, 1))
        #
        # counts, steps = histogram(total_outer_abs_sampled, bins=100_000, density=True)
        # plt.stairs(counts.cpu(), steps.cpu(), label='gamma')

        # theoretical_output = sum_function(steps, s)
        # plt.plot(steps, theoretical_output)

        # theoretical_c = difference(steps, dl + 1, s)
        # plt.plot(steps, theoretical_c)
        # plt.xlim(-3, 3)

        plt.legend()

        plt.show()


def check_outer_sum_specific_dl(nl_=2, ax=None):
    N = 500_000_000

    dl_ = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    s = 1

    show = False
    if ax is None:
        fig, ax = plt.subplots()
        fig: Figure
        show = True

    for dl_step in range(2, dl_ + 2, 2):

        total_outer_abs = torch.zeros((N, 1), device=device)
        for nl in range(nl_):

            total = torch.zeros((N, 1), device=device)

            for dl in range(dl_step):
                w1 = torch.normal(0, s, size=(N, 1), device=device)
                w2 = torch.normal(0, s, size=(N, 1), device=device)

                total += w1 * w2

            total_outer_abs += torch.abs(total)

        counts, steps = histogram(total_outer_abs, bins=100_000, density=True)
        ax.stairs(counts.cpu(), steps.cpu(), label=str(dl_step))

    ax.set_title(f"n = {nl_}")
    ax.set_xlim((0, 10))
    ax.minorticks_on()
    ax.grid(which='both')
    ax.legend()

    if show:
        fig.tight_layout()

        plt.show()


def outer_multi_n_plots():
    fig, axes = plt.subplots(2, 2)

    flattened = [item for row in axes for item in row]

    for i, ax in enumerate(flattened):
        print(i)
        check_outer_sum_specific_dl(i + 1, ax)

    fig.tight_layout()
    plt.savefig("sample_output.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    # check_inner_sum(abs=True)
    # check_outer_sum()
    # check_outer_sum_specific_dl()

    outer_multi_n_plots()
