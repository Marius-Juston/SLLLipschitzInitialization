"""Exact Gaussian normalized variance and independent Monte Carlo validation."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln, erfcx, roots_genlaguerre


def mean_field(d, n):
    return 1 / (d + (n - 1) * 2 / math.sqrt(math.pi) * math.exp(gammaln((d + 1)/2) - gammaln(d/2)))


def gaussian_variance_adaptive(d, n):
    """Independent nested adaptive quadrature, useful near d=1,2."""
    logc = (1-d/2)*math.log(2)-gammaln(d/2)
    def transform(t):
        if t <= 1:
            return quad(lambda r: math.exp(logc+d*math.log(r)-r*r/2-t*r) if r else 0.,
                        0,np.inf,epsabs=1e-12)[0]
        # Rescale to keep the mass resolved when t is large.
        return math.exp(logc-(d+1)*math.log(t))*quad(
            lambda u: math.exp(d*math.log(u)-u-u*u/(2*t*t)) if u else 0.,
            0,np.inf,epsabs=1e-12)[0]
    return quad(lambda t: transform(t)*erfcx(t/math.sqrt(2))**(n-1),
                0,np.inf,epsabs=1e-10,epsrel=1e-9)[0]/d


def gaussian_variance(d, n, order=128):
    """Deterministic quadrature of E[R/(R+H)]/d; order controls chi quadrature.

    R^2/2 is Gamma(d/2,1). Integrate the half-normal Laplace transform
    exactly in the outer integrand, using scaled erfc to avoid overflow.
    """
    if d < 1 or n < 1:
        raise ValueError('Dimensions must be positive')
    if n == 1:
        return 1 / d
    if d == 1:  # exchangeable |G_i| / sum |G_j|
        return 1 / n
    if d <= 4:
        return gaussian_variance_adaptive(d,n)
    nodes, weights = roots_genlaguerre(order, d/2 - 1)
    weights = weights / weights.sum()
    r = np.sqrt(2 * nodes)
    value = quad(lambda t: np.dot(weights, r * np.exp(-t*r)) * erfcx(t/math.sqrt(2))**(n-1),
                 0, np.inf, epsabs=1e-10, epsrel=1e-8)[0]
    return value / d


def sample_weights(rng, shape, distribution):
    if distribution == 'gaussian':
        return rng.normal(size=shape)
    if distribution == 'laplace':
        return rng.laplace(scale=1/math.sqrt(2), size=shape)
    return rng.uniform(-math.sqrt(3), math.sqrt(3), size=shape)


def validate(samples=20000, seed=813):
    rng = np.random.default_rng(seed)
    results = []
    for distribution in ['gaussian', 'laplace', 'uniform']:
        for d, n in [(1, 2), (2, 4), (8, 8), (16, 4), (32, 32), (64, 16)]:
            values = []
            for start in range(0, samples, 256):
                w = sample_weights(rng, (min(256, samples-start), d, n), distribution)
                denominator = np.abs(np.swapaxes(w, 1, 2) @ w).sum(axis=2)
                # One statistic per independent matrix: entries are NOT independent replicates.
                values.extend((w[:,:,0]**2 / denominator[:,None,0]).mean(axis=1))
            values = np.asarray(values)
            row = dict(distribution=distribution, d=d, n=n, samples=samples,
                       estimate=float(values.mean()), se=float(values.std(ddof=1)/math.sqrt(samples)),
                       distribution_free_upper=1/d)
            if distribution == 'gaussian':
                row.update(exact_quadrature=gaussian_variance(d,n), mean_field=mean_field(d,n),
                           quadrature_order_check=abs(gaussian_variance(d,n,256)-gaussian_variance(d,n)))
            results.append(row)
    return dict(seed=seed, results=results)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=int, default=20000)
    p.add_argument('--output', type=Path, default=Path('artifacts/math-validation.json'))
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(validate(args.samples), indent=2)+'\n')
    print(args.output)


def product_cf(t, distribution):
    """Characteristic function of XY for independent unit-variance factors."""
    from scipy.special import sici
    t=abs(float(t))
    if t==0:return 1.
    if distribution=='gaussian':return (1+t*t)**-.5
    if distribution=='uniform':return sici(3*t)[0]/(3*t)
    if distribution=='laplace':
        if t<.01:
            return quad(lambda u:math.exp(-u)/(1+(t*u/2)**2),0,np.inf,epsabs=1e-13)[0]
        x=2/t;si,ci=sici(x)
        return x*(ci*math.sin(x)+(math.pi/2-si)*math.cos(x))
    raise ValueError(distribution)


def product_sum_absolute_moment(n, distribution):
    # Compactify t in [0,infinity) to u in [0,1]; cancellation handled with expm1.
    def integrand(u):
        if u<1e-5:return n/2
        if u==1:return 1.
        phi=product_cf(u/(1-u),distribution)
        return -math.expm1(n*math.log(min(1.,phi)))/(u*u)
    return 2/math.pi*quad(integrand,0,1,epsabs=2e-8,limit=500)[0]
