import matplotlib.pyplot as plt
import torch
from torch.distributions.beta import Beta

from distribution import histogram

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    s = 3
    N = 10_000

    for i in range(2, 10):
        dl = i

        W = torch.normal(0, s ** 2, size=(N, dl), device=device)

        normalization = W.square().sum(dim=1).sqrt().reshape(N, 1)

        transformed = W / normalization

        mean = transformed.mean()
        variance = transformed.var()

        freq, count = histogram(transformed ** 2, 100, density=True)

        plt.stairs(freq.cpu(), count, label='actual')

        distri = Beta(torch.tensor([1 / 2]), torch.tensor([(dl - 1) / 2]))
        sample = distri.sample((N,))

        freq, count = histogram(sample, 100, density=True)

        print(mean, variance, distri.mean)

        plt.stairs(freq.cpu(), count, label='theoretical')

        plt.legend()

        plt.savefig(f'../../figs/upper_bound_distribution/upper_bound_distribution_{dl}.png', dpi=600)

        plt.close('all')
