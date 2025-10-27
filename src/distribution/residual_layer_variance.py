import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch.nn import ReLU

from distribution import histogram
from models.layers import safe_inv

torch.set_float32_matmul_precision('high')


def forward(x, W, b, act):
    # count, freq = histogram(W, 100, density=True)
    # plt.stairs(count.cpu(), freq.cpu())
    # plt.show()

    t = torch.abs(torch.einsum('bik,bjk -> bij', W, W)).sum(2)

    # mask = (torch.abs(t) > 1e-5).any(dim=1)
    #
    #
    # t = t[mask]
    # x = x[mask]
    # W = W[mask]
    # b = b[mask]

    # count, freq = histogram(t, 100, density=True)
    # plt.stairs(count.cpu(), freq.cpu())
    # plt.show()
    #
    # count, freq = histogram(torch.reciprocal(t), 100, density=True)
    # plt.stairs(count.cpu(), freq.cpu())
    # plt.show()

    res = torch.einsum('bki,bi -> bk', W, x) + b

    # count, freq = histogram(res, 100, density=True)
    # plt.stairs(count.cpu(), freq.cpu())
    # plt.show()

    act_res = act(res)

    # count, freq = histogram(act_res, 100, density=True)
    # plt.stairs(count.cpu(), freq.cpu())
    # plt.show()

    res = torch.reciprocal(t) * act_res

    # count, freq = histogram(res, 100, density=True)
    # plt.stairs(count.cpu(), freq.cpu())
    # plt.show()

    res = 2 * torch.einsum('bik,bi -> bk', W, res)

    # count, freq = histogram(res, 100, density=True)
    # plt.stairs(count.cpu(), freq.cpu())
    # plt.show()

    out = x - res

    return out, t


SQRT_PI = np.sqrt(torch.pi)


def gamma_div(num, denom):
    return (torch.lgamma(num) - torch.lgamma(denom)).exp()


def D_sqaure(s_w, nl, dl):
    D1 = 3 * nl ** 2 * s_w ** 2
    D2 = (dl - 1) * nl * s_w ** 2

    gamma_val = gamma_div((nl + 1) / 2., nl / 2.)

    D3 = 2 * (dl - 1) * nl * s_w ** 2 * 2 / SQRT_PI * gamma_val
    D4 = (dl - 1) * (dl - 2) * s_w ** 2 * 4 / torch.pi * gamma_val ** 2

    D = D1 + D2 + D3 + D4

    return D


def C0(s_w, s_b, nl, dl):
    N1 = s_w * s_b

    N = N1
    D = D_sqaure(s_w, nl, dl)

    return 2 * dl * N / D


def C1(s_w, s_b, nl, dl):
    N2 = (nl + 2) * s_w ** 2

    N = N2

    D = D_sqaure(s_w, nl, dl)

    return 2 * dl * N / D


def C1_special(s_w, s_b, nl, dl, D):
    N = (nl * 6) * s_w ** 2

    return 2 * dl * N / D


def variance_product(zero_mean=True, dl_scale: float = 10, N=10_000, nl_max=100, Nmin=100, samples=1, histograms=False,
                     super_histogram=True, save=True, show=True):
    variances = []
    means = []

    t_2 = []

    dtype = torch.float64

    nls = np.arange(1, nl_max + 1)

    # dls = np.maximum(1, ((nls + 1) * dl_scale + 1).astype(nls.dtype))
    dls = np.maximum(1, (nls * dl_scale).astype(nls.dtype))
    # dls = np.full_like(dls, 1)

    s_w = 1
    s_b = 1

    act = ReLU()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if super_histogram:
        fig, super_ax = plt.subplots()

    for nl, dl in zip(nls, dls):
        if histograms:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig: Figure
            ax1: Axes
            ax2: Axes

        sample_vars = []
        sample_means = []
        sample_t_2 = []

        batch_size = int(max(N / (dl * nl), Nmin))

        for sample in range(samples):

            W = torch.normal(0., float(np.sqrt(s_w)), size=(batch_size, dl, nl), device=device, dtype=dtype)
            b = torch.normal(0., float(np.sqrt(s_b)), size=(batch_size, dl), device=device, dtype=dtype)
            # b = torch.zeros(size=(batch_size, dl), device=device, dtype=dtype)

            if zero_mean:
                # x = torch.normal(0., 1., size=(batch_size, nl), device=device, dtype=dtype)
                x = torch.zeros(size=(batch_size, nl), device=device, dtype=dtype)
            else:
                x = torch.normal(3., 5., size=(batch_size, nl), device=device, dtype=dtype)

            yl, t = forward(x, W, b, act)

            if not torch.isreal(yl).all():
                continue

            yl_m = yl[:, :, None]
            yl_m_t = yl_m.transpose(1, 2)

            yy_t = torch.diagonal(torch.bmm(yl_m, yl_m_t), dim1=-2, dim2=-1)

            sample_vars.append(yl.var().item())
            sample_means.append(yy_t.mean().item())
            sample_t_2.append(torch.square(t).mean().item())

            print(sample_vars)

            if histograms or super_histogram:
                count, freq = histogram(yl, density=True)

                if histograms:
                    ax1.stairs(count.cpu(), freq.cpu())

                if sample == 0 and super_histogram:
                    super_ax.stairs(count.cpu(), freq.cpu())

            r = torch.cuda.memory_reserved(0)

            if r >= 23.9e9:
                print("FREEING CACHED MEMORY")
                torch.cuda.empty_cache()

        if histograms:
            ax2.hist(sample_vars)
            plt.show()

        sample_var_mean = np.mean(sample_vars)
        sample_mean_mean = np.mean(sample_means)
        sample_t_2_mean = np.mean(sample_t_2)

        variances.append(sample_var_mean)
        means.append(sample_mean_mean)
        t_2.append(sample_t_2_mean)

        print(nl, batch_size * (dl * nl), batch_size, dl * nl, N / (dl * nl), sample_var_mean, sample_mean_mean,
              sample_t_2)

    torch.cuda.empty_cache()

    nls = torch.from_numpy(nls)
    dls = torch.from_numpy(dls)
    t_2 = torch.tensor(t_2)

    variances = torch.tensor(variances)

    print(variances)

    if super_histogram:
        plt.show()

    # plt.plot(nls, t_2, label='t2')
    # plt.plot(nls, D_sqaure(s_w, nls, dls), label='D2')
    # plt.legend()
    # print(t_2)
    # plt.show()

    difference = variances * t_2 / s_w ** 2 / dls / 2 / (dls - 2)

    print(nls, dls)
    print(difference)
    print(difference / (dls - 2))

    # plt.plot(nls, difference, label='variance')
    # plt.show()

    plt.plot(nls, variances, label='variance')
    # plt.plot(nls, means)

    # bias_variances_simulated = bias_simulation(s_w, s_b, nls, dls)
    # print(bias_variances_simulated)
    # plt.plot(nls, bias_variances_simulated, label='C0_simulated')

    print(variances)
    plt.plot(nls, C0(s_w, s_b, nls, dls), label='C0')
    # plt.plot(nls, C1(s_w, s_b, nls, dls), label='C1')
    # plt.plot(nls, C1_special(s_w, s_b, nls, dls, t_2), label='C1 special')

    # plt.yscale('log')
    plt.grid()
    plt.tight_layout()

    if show:
        plt.legend()
        plt.show()

    if save:
        s = torch.zeros((nls.size(0), 2))
        s[:, 0] = s_w
        s[:, 1] = s_b

        np.savetxt('residual_variances_2.csv', torch.column_stack((nls, dls, s, variances)), delimiter=',',
                   fmt=['%i', '%i', '%.18e', '%.18e', '%.18e'])


def compute_t_2(q_, weight):
    q = torch.exp(q_)
    q_inv = torch.exp(q_)
    t = torch.abs(torch.einsum('i,ik,kj,j -> ij', q_inv, weight, weight.T, q)).sum(1)
    t = safe_inv(t)
    return t


def forward_2(weight, bias, x, q_):
    t = compute_t_2(q_, weight)
    res = F.linear(x, weight, bias)
    res = t * ReLU()(res)
    res = 2 * F.linear(res, weight.T)
    out = x - res
    return out


def computation_verification():
    dl = 2
    nl = 4

    W = torch.arange(nl * dl).reshape(1, dl, nl).to(dtype=torch.float32) + 1
    b = torch.arange(dl).reshape(1, dl).to(dtype=torch.float32) + dl * nl + 1
    x = torch.arange(nl).reshape(1, nl).to(dtype=torch.float32) + dl * nl + dl + 1

    print(W, b, x)

    q = torch.zeros_like(b)

    out = forward(x, W, b, ReLU())
    out2 = forward_2(W[0], b[0], x[0], q[0])

    print(out, out2)


def C0_verification():
    nls = torch.arange(1, 100)
    dls = torch.arange(1, 100)

    print(C0(1, 1, nls, dls).tolist())


if __name__ == '__main__':
    # C0_verification()

    # computation_verification()
    #
    # variance_product(dl_scale=10, Nmin=100, samples=1000, super_histogram=True, nl_max=200, histograms=False)

    for i in range(2, 10 + 1):
        variance_product(dl_scale=i, Nmin=1_0_000, samples=100, super_histogram=False, nl_max=10, histograms=False,
                         save=False,
                         show=True)
    # variance_product(dl_scale=10, Nmin=100, samples=10, super_histogram=False, nl_max=100, histograms=False, save=False)
