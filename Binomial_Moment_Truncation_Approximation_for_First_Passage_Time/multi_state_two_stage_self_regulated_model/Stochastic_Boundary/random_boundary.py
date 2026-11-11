"""Jacobi、双端反射 OU 和双端反射布朗，以及可配置的首达判据。"""
from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh
from scipy.special import eval_jacobi, roots_jacobi
from scipy.stats import truncnorm


def protein_threshold(m, n, x):
    return n - x


@dataclass(frozen=True)
class RandomBoundary:
    kind: str
    lower: float = 3.5
    upper: float = 7.5
    sigma: float = 1.
    kappa: float = 1.
    theta: float = .5  # Jacobi 用区间比例；OU 用区间位置的比例
    criterion: Callable = protein_threshold

    def __post_init__(self):
        if (self.kind not in ('brownian', 'jacobi', 'ou') or self.lower >= self.upper
                or self.sigma <= 0 or self.kappa <= 0 or not 0 < self.theta < 1):
            raise ValueError('Invalid boundary parameters')

    def g(self, m, n, x):
        """计算 g >= 0 的判据，支持 NumPy 数组和广播。"""
        return np.broadcast_to(self.criterion(m, n, x), np.broadcast(m, n, x).shape)

    def initial(self, rng, paths):
        if self.kind == 'brownian':
            y = rng.random(paths)
        elif self.kind == 'jacobi':
            a = 2 * self.kappa * self.theta / self.sigma**2
            b = 2 * self.kappa * (1-self.theta) / self.sigma**2
            y = rng.beta(a, b, paths)
        else:
            center = self.lower + self.theta * (self.upper-self.lower)
            spread = self.sigma / np.sqrt(2 * self.kappa)
            return truncnorm.rvs((self.lower-center)/spread, (self.upper-center)/spread,
                                 loc=center, scale=spread, size=paths, random_state=rng)
        return self.lower + (self.upper-self.lower) * y

    def advance(self, x, elapsed, rng):
        width = self.upper-self.lower
        if self.kind == 'jacobi':
            y = (x-self.lower)/width
            y += self.kappa*(self.theta-y)*elapsed
            y += self.sigma*np.sqrt(np.clip(y*(1-y), 0, None)*elapsed)*rng.normal(size=x.shape)
            return self.lower + width*np.clip(y, 0, 1)
        if self.kind == 'ou':
            center = self.lower + self.theta*width
            x = x + self.kappa*(center-x)*elapsed
        x = x-self.lower + self.sigma*np.sqrt(elapsed)*rng.normal(size=x.shape)
        return self.lower + width - np.abs(x % (2*width)-width)

    def basis(self, count, quadrature):
        width = self.upper-self.lower
        degrees = np.arange(count)
        if self.kind == 'jacobi':
            a = 2*self.kappa*self.theta/self.sigma**2
            b = 2*self.kappa*(1-self.theta)/self.sigma**2
            nodes, weights = roots_jacobi(quadrature, b-1, a-1)
            weights /= weights.sum()
            x = self.lower + (nodes+1)*width/2
            raw = np.column_stack([eval_jacobi(j, b-1, a-1, nodes) for j in degrees])
            phi = raw/np.sqrt(np.sum(weights[:, None]*raw**2, axis=0))
            eigenvalues = self.kappa*degrees + .5*self.sigma**2*degrees*(degrees-1)
            return x, weights, phi, eigenvalues
        nodes, weights = leggauss(quadrature)
        x = self.lower + (nodes+1)*width/2
        raw = np.cos(np.pi*(nodes[:, None]+1)*degrees/2)
        derivative = -np.pi*degrees[None, :]/width*np.sin(np.pi*(nodes[:, None]+1)*degrees/2)
        if self.kind == 'brownian':
            phi = raw*np.where(degrees == 0, 1., np.sqrt(2.))[None, :]
            return x, weights/2, phi, .5*self.sigma**2*(np.pi*degrees/width)**2
        center = self.lower + self.theta*width
        weights *= np.exp(-self.kappa*(x-center)**2/self.sigma**2)
        weights /= weights.sum()
        mass = raw.T @ (weights[:, None]*raw)
        stiffness = .5*self.sigma**2*derivative.T @ (weights[:, None]*derivative)
        eigenvalues, vectors = eigh(stiffness, mass)
        phi = raw @ vectors
        phi[:, 0] = 1.  # 归一常数基，零阶边界系数即总质量
        return x, weights, phi, eigenvalues


BrownianBoundary = lambda **kwargs: RandomBoundary('brownian', **kwargs)
