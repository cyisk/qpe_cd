import numpy as np

from typing import Callable, Sequence, Tuple
from numpy.typing import NDArray

DEFAULT_BASIS_FUNCS_NUMPY = [
    lambda y: np.ones_like(y),
    lambda y: y,
    lambda y: y**3,
    lambda y: np.tanh(y),
]


def qpe_k(
    x1: NDArray,
    x2: NDArray,
    basis_funcs: Sequence[Callable] = DEFAULT_BASIS_FUNCS_NUMPY,
) -> Tuple[Callable, float, Callable, float]:
    if x1.ndim <= 1 or x2.ndim <= 1:
        x1, x2 = x1[:, None], x2[:, None]

    def qpe(X, y, Xq, yq, h_x=None, h_y=None):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        N, d = X.shape
        if h_x is None:
            h_x = X.std(0) * (4 / (d + 2))**(1 / (d + 4)) * N**(-1 / (d + 4))
        else:
            h_x = np.full(d, h_x)
        if h_y is None:
            h_y = y.std() * (4 / 3)**0.2 * N**(-0.2)
        diff_x = Xq[:, None, :] - X[None, :, :]
        Kx = np.exp(-np.sum(diff_x**2 / (2 * h_x**2), axis=-1))
        S = sigmoid((yq[:, None] - y[None, :]) / h_y)
        D = np.sum(Kx, axis=1, keepdims=True)
        N_ = np.sum(Kx * S, axis=1, keepdims=True)
        dK = diff_x / (h_x**2) * Kx[:, :, None]
        dN = np.sum(dK * S[:, :, None], axis=1)
        dD = np.sum(dK, axis=1)
        dF = (dN * D - N_ * dD) / (D * D)
        Sy = S * (1 - S) / h_y
        p_num = np.sum(Kx * Sy, axis=1)
        p_hat = p_num / D.squeeze(1)
        return dF / p_hat[:, None]

    def lsquare_test(X, y, Xq, yq, h, basis_funcs):
        B = np.stack([bf(yq) for bf in basis_funcs], axis=1)
        n_y = yq.shape[0]
        G_inv = np.linalg.pinv(B.T @ B)
        Xq_rep = np.repeat(Xq, n_y, axis=0)
        yq_rep = np.tile(yq, Xq.shape[0])
        H = h(X, y, Xq_rep, yq_rep).reshape(Xq.shape[0], n_y)
        A = (H @ B) @ G_inv
        R = H - A @ B.T
        return np.linalg.norm(R, axis=1)

    def qpek_score(x, y):
        Xg, yg = np.linspace(-2.5, 2.5, 20), np.linspace(-2.5, 2.5, 20)
        return -lsquare_test(x, y[:, 0], Xg[:, None], yg, qpe,
                             basis_funcs).mean()

    def wrapper(transpose=False):

        def call_cv(x: NDArray, y: NDArray):
            if transpose:
                x, y = y, x
                X, Y = x2, x1
            else:
                X, Y = x1, x2
            return qpe(X, Y[:, 0], x, y[:, 0])

        return call_cv

    s12 = qpek_score(x1, x2)
    s21 = qpek_score(x2, x1)
    return wrapper(), s12, wrapper(transpose=True), s21
