import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def laplace(nl, dl):
    return 2 / (2 * dl ** (1 / 2) * (nl - 1) + (dl - 1) * dl + 2)


def normal(nl, dl):
    return 1 / (dl ** (1 / 2) * (dl ** (1 / 2) + nl - 1))


def uniform(nl, dl):
    return (1 + dl) / (dl ** (3 / 2) * (nl - 1) + dl ** (1 / 2) * (nl - 1) + 4 * dl - 2)


if __name__ == '__main__':
    N = 100

    nl = dl = np.linspace(1, 10, N)
    xx, yy = np.meshgrid(nl, dl)

    funcs = [laplace, normal, uniform]
    colors = ['r', 'g', "b"]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax: Axes3D

    for f, c in zip(funcs, colors):
        label = f.__name__

        zz = f(xx, yy)

        surf = ax.plot_surface(xx, yy, zz.T, edgecolor='k', label=label, linewidth=0.2)

    ax.legend()
    ax.set_xlabel("$d_l$")
    ax.set_ylabel("$n_l$")
    ax.set_zlabel("Var[$\\bar{w}_l$]")

    plt.tight_layout()

    plt.savefig("../../figs/GeneralizedVariance.png", dpi=300)
    plt.show()
