import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from causallearn.utils.FastKCI.FastKCI import FastKCI_CInd as cci, FastKCI_UInd as uci

from typing import Tuple
from numpy.typing import NDArray


def stein_score(
    X: NDArray,
    eta_G: float = 0.001,
) -> NDArray:
    n, _ = X.shape
    X_diff = np.expand_dims(X, axis=1) - X
    D = np.linalg.norm(X_diff, axis=2).flatten()
    D_nonzeros = D[D > 0]
    s = np.median(D_nonzeros) if np.any(D_nonzeros) else 1
    K = rbf_kernel(X, gamma=1 / (2 * s**2)) / s
    X_diff = np.expand_dims(X, axis=1) - X
    nablaK = -np.einsum("kij,ik->kj", X_diff, K) / s**2
    G = np.matmul(np.linalg.inv(K + eta_G * np.eye(n)), nablaK)
    return G


def fisher_info(
    X: NDArray,
    eta_G: float = 0.001,
) -> float:
    score = stein_score(X, eta_G=eta_G)
    return np.var(score, axis=0)


def fico(
    X: NDArray,
    eta_G: float = 0.001,
    alpha: float = 0.05,
    return_graph: bool = False,
) -> NDArray | Tuple[NDArray, NDArray]:
    n, d = X.shape
    R, order = np.arange(n), []
    Xt = np.copy(X)
    for _ in range(d, 1, -1):
        l = np.argmin(fisher_info(Xt, eta_G=eta_G))
        order.append(R[l])
        R = np.concat([R[:l], R[l + 1:]], axis=0)
        Xt = np.concat([Xt[:, :l], Xt[:, l + 1:]], axis=1)
    order.append(R[0])
    order = np.array(order)[::-1]
    if not return_graph:
        return order

    A = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            it, jt = order[i], order[j]
            Xi, Xj = X[:, it:it + 1], X[:, jt:jt + 1]
            ct = np.concat([order[:i], order[i + 1:j]], axis=0)
            if len(ct) > 0:
                p, _ = cci().compute_pvalue(Xi, Xj, X[:, ct])
            else:
                p, _ = uci().compute_pvalue(Xi, Xj)
            if p < alpha:
                A[it, jt] = 1.0
                print(it, jt)
    return order, A
