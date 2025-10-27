import matplotlib.pyplot as plt
import torch
from torch.distributions import Chi2


def expected_value(n):
    n = torch.tensor(n)

    const = 4 / torch.sqrt(torch.tensor(torch.pi))

    num = torch.lgamma((n + 1) / 2.)
    denom = torch.lgamma(n / 2.)

    print(num)
    print(denom)

    return const * (num - denom).exp()


if __name__ == '__main__':
    N = 1000000

    s = 1

    nl = 5

    X = torch.normal(0, std=s, size=(N, nl))
    Y = torch.normal(0, std=s, size=(N, nl))

    Z = torch.abs((X * Y).sum(dim=1))

    sampled_z_mean = Z.mean()

    print(Z.var(), sampled_z_mean)

    count, bins = torch.histogram(Z, 100)
    plt.stairs(count, bins, label='sampled')


    m = Chi2(torch.tensor([nl]))
    Q = m.sample((N,))
    R = m.sample((N,))

    Z = s ** 2 * (1 / 2 * Q - 1 / 2 * R)
    Z = torch.abs(Z)

    theoretical_z_mean = Z.mean()

    theoretical_expected = s ** 2 / 2 * expected_value(nl)

    print(Z.var(), Z.mean(), theoretical_expected)

    count, bins = torch.histogram(Z, 100)
    plt.stairs(count, bins, label='theoretical')

    max_v = count.max()

    plt.plot([sampled_z_mean, sampled_z_mean], [0, max_v], label='sample_mean', c='r')
    plt.plot([sampled_z_mean, sampled_z_mean], [0, max_v], label='theoretical_mean', c='g')
    plt.plot([sampled_z_mean, sampled_z_mean], [0, max_v], label='calculated_mean', c='b')

    plt.legend()
    plt.show()
