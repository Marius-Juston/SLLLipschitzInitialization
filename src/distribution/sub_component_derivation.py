import matplotlib.pyplot as plt
import numpy as np
import torch

from distribution.distribution import histogram

if __name__ == '__main__':
    s_w = 2

    sqrt_s_w = np.sqrt(s_w)

    N = 100000

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    hist = False

    means_s2 = []

    ns = torch.arange(1, 100)

    for n in ns:
        w_a = torch.normal(0, sqrt_s_w, size=(n, N), device=device)
        w_b = torch.normal(0, sqrt_s_w, size=(n, N), device=device)

        prod = (w_a * w_b)

        s = prod.sum(dim=0).abs()
        expectation_1 = s.mean()

        theoretical_expectation_1 = 4 / np.sqrt(torch.pi) * torch.exp(
            torch.special.gammaln(torch.tensor((n + 1) / 2)) - torch.special.gammaln(torch.tensor(n / 2))) * s_w / 2

        print(expectation_1, theoretical_expectation_1)

        s_2 = torch.square(s)

        if hist:
            fig, (ax1, ax2) = plt.subplots(1, 2)

            count, freq = histogram(s, 100)
            ax1.stairs(count.cpu(), freq.cpu())

            count, freq = histogram(s_2, 100)
            ax2.stairs(count.cpu(), freq.cpu())
            plt.show()

        expectation_2 = s_2.mean()

        means_s2.append(expectation_2.item())

        theoretical_expectation_2 = n * s_w ** 2

        print(expectation_2, theoretical_expectation_2)

    plt.plot(ns, means_s2, label='actual')
    plt.plot(ns, ns * s_w ** 2, label='theoretical')
    plt.show()
