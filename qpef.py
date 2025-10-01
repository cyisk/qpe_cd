import numpy as np
import torch as th
import zuko as zk
from copy import deepcopy
from torch.autograd.functional import jvp

from typing import Any, Callable, Sequence, Dict, Type, Tuple
from numpy.typing import NDArray
from torch.types import Tensor, Device


def to_numpy(x: Tensor | NDArray, ) -> NDArray:
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def to_torch(
    x: NDArray | Tensor,
    device: Device = None,
) -> Tensor:
    if isinstance(x, th.Tensor):
        return x
    return th.from_numpy(x).float().to(device)


def df_dx(
    f: Callable,
    x: Tensor,
    create_graph: bool = False,
) -> Tensor:
    return jvp(f, (x, ), (th.ones_like(x), ), create_graph=create_graph)[1]


class AffineConditionalTransform(th.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.hyper = zk.nn.MLP(1, 2, **kwargs)

    def t(self, x: th.Tensor) -> th.distributions.Transform:
        theta = self.hyper(x[..., None])
        shift = theta[..., 0]
        scale = theta[..., 1]
        return zk.transforms.MonotonicAffineTransform(
            shift=shift,
            scale=scale,
        )


class RQSConditionalTransform(th.nn.Module):

    def __init__(self, bins: int = 8, **kwargs):
        super().__init__()
        self.bins = bins
        self.hyper = zk.nn.MLP(1, 3 * bins - 1, **kwargs)

    def t(self, x: th.Tensor) -> th.distributions.Transform:
        theta = self.hyper(x[..., None])
        widths = theta[..., 0:self.bins]
        heights = theta[..., self.bins:2 * self.bins]
        derivatives = theta[..., 2 * self.bins:3 * self.bins - 1]
        return zk.transforms.MonotonicRQSTransform(
            widths=widths,
            heights=heights,
            derivatives=derivatives,
        )


class MNNConditionalTransform(th.nn.Module):

    def __init__(self, signal: int = 16, **kwargs):
        super().__init__()
        self.signal = signal
        self.hyper = zk.nn.MLP(1, signal, **kwargs)
        self.mnn = zk.flows.neural.MNN(signal=signal, **kwargs)

    def t(self, x: th.Tensor) -> th.distributions.Transform:
        theta = self.hyper(x[..., None])
        signal = theta[..., 0:self.signal]
        return self.mnn(signal)


class UMNNConditionalTransform(th.nn.Module):

    def __init__(self, signal: int = 16, **kwargs):
        super().__init__()
        self.signal = signal
        self.hyper = zk.nn.MLP(1, signal + 1, **kwargs)
        self.umnn = zk.flows.neural.UMNN(signal=signal, **kwargs)

    def t(self, x: th.Tensor) -> th.distributions.Transform:
        theta = self.hyper(x[..., None])
        signal = theta[..., 0:self.signal]
        constant = theta[..., self.signal]
        return self.umnn(signal, constant)


class ConditionalFlow(th.nn.Module):

    def __init__(
        self,
        transform_cls: Type = UMNNConditionalTransform,
        transforms: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.transforms = transforms
        self.lazy_transforms = th.nn.ModuleList(
            [transform_cls(**kwargs) for i in range(transforms)])
        self.lazy_base = zk.lazy.UnconditionalDistribution(
            th.distributions.Normal,
            th.zeros(th.Size()),
            th.ones(th.Size()),
            buffer=True,
        )

    def base(self) -> th.distributions.Distribution:
        return self.lazy_base()

    def t(self, x: th.Tensor) -> th.distributions.Transform:
        return zk.transforms.ComposedTransform(
            *[self.lazy_transforms[i].t(x) for i in range(self.transforms)])

    def log_prob(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        t = self.t(x)
        u = t(y)
        log_pu = self.base().log_prob(u)
        ladj = t.log_abs_det_jacobian(y, u)
        return log_pu + ladj


DEFAULT_BASIS_FUNCS_TORCH = [
    lambda y: th.ones_like(y),
    lambda y: y,
]


class PolynomialBasisNeuralQPE(th.nn.Module):

    def __init__(
        self,
        degree: int = 1,
        hidden_features: Sequence[int] = (64, 64),
        activation: th.nn.Module = th.nn.Tanh,
    ):
        super().__init__()
        self.degree = degree
        self.coeff_net = zk.nn.MLP(
            in_features=1,
            out_features=degree + 1,
            hidden_features=hidden_features,
            activation=activation,
        )

    def forward(self, x, y):
        C = self.coeff_net(x[..., None])
        poly = C[..., 0]
        for i in range(1, self.degree + 1):
            poly = C[..., i] + y * poly
        return poly


class FixedBasisNeuralQPE(th.nn.Module):

    def __init__(
        self,
        basis_funcs: Sequence[Callable] = DEFAULT_BASIS_FUNCS_TORCH,
        hidden_features: Sequence[int] = (64, 64),
        activation: th.nn.Module = th.nn.Tanh,
    ):
        super().__init__()
        self.basis_funcs = basis_funcs
        self.coeff_net = zk.nn.MLP(
            in_features=1,
            out_features=len(basis_funcs),
            hidden_features=hidden_features,
            activation=activation,
        )

    def forward(self, x, y):
        C = self.coeff_net(x[..., None])
        B = th.stack([b(y) for b in self.basis_funcs], dim=-1)
        return th.einsum("...i,...i->...", C, B)


class LowrankNeuralQPE(th.nn.Module):

    def __init__(
        self,
        rank: int = 2,
        hidden_features: Sequence[int] = (64, 64),
        activation: th.nn.Module = th.nn.Tanh,
    ):
        super().__init__()
        self.x_net = zk.nn.MLP(
            in_features=1,
            out_features=rank,
            hidden_features=hidden_features,
            activation=activation,
        )
        self.y_net = zk.nn.MLP(
            in_features=1,
            out_features=rank,
            hidden_features=hidden_features,
            activation=activation,
        )

    def forward(self, x, y):
        X = self.x_net(x[..., None])
        Y = self.y_net(y[..., None])
        return th.sum(X * Y, dim=-1)


class UnconstrainedNeuralQPE(th.nn.Module):

    def __init__(
        self,
        hidden_features: Sequence[int] = (64, 64),
        activation: th.nn.Module = th.nn.Tanh,
    ):
        super().__init__()
        self.hyper_net = zk.nn.MLP(
            in_features=2,
            out_features=1,
            hidden_features=hidden_features,
            activation=activation,
        )

    def forward(self, x, y):
        xy = th.stack([x, y], dim=-1)
        return self.hyper_net(xy)


def qpe_f(
    x1: NDArray,
    x2: NDArray,
    transform_cls: Type = UMNNConditionalTransform,
    transforms: int = 1,
    neural_qpe_cls: Type = FixedBasisNeuralQPE,
    neural_qpe_kwargs: Dict[str, Any] = {
        'basis_funcs': DEFAULT_BASIS_FUNCS_TORCH,
    },
    epochs: int = 1000,
    learning_rate: float = 0.01,
    weight_decay: float = 0.001,
    eta_min: float = 1e-6,
    hidden_features: Sequence[int] = (100, 100),
    activation: th.nn.Module = th.nn.Tanh,
) -> Tuple[Callable, float, Callable, float]:
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    x1, x2 = to_torch(x1, device)[:, 0], to_torch(x2, device)[:, 0]

    def train_model(model: th.nn.Module, loss_func):
        optimizer = th.optim.Adam(
            params=model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=epochs,
            eta_min=eta_min,
        )
        model.train()
        best_state, min_loss = deepcopy(model.state_dict()), th.inf
        for i in range(epochs):
            loss = loss_func(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if loss < min_loss:
                best_state, min_loss = deepcopy(model.state_dict()), loss
        model.load_state_dict(best_state)
        model.eval()
        return model, min_loss

    def train_flow(x, y):
        flow = ConditionalFlow(
            transform_cls=transform_cls,
            transforms=transforms,
            hidden_features=hidden_features,
            activation=activation,
        ).to(device)
        flow, loss = train_model(
            model=flow,
            loss_func=lambda flow: -flow.log_prob(x, y).mean(),
        )
        return flow, -loss.item()

    def qpe(flow, x, y):
        du_dx = df_dx(lambda x_: flow.t(x_)(y), x)
        du_dy = df_dx(lambda y_: flow.t(x)(y_), y)
        return -du_dx / du_dy

    def train_nqpe(x, y, flow):
        nqpe = neural_qpe_cls(
            hidden_features=hidden_features,
            activation=activation,
            **neural_qpe_kwargs,
        ).to(device)
        qpe_hat = qpe(flow, x, y).detach()
        nqpe, loss = train_model(
            model=nqpe,
            loss_func=lambda nqpe: ((qpe_hat - nqpe(x, y))**2).mean(),
        )
        return nqpe, -loss.item()

    def wrapper(flow, transpose=False):

        def call_cv(x: NDArray, y: NDArray):
            x, y = to_torch(x, device), to_torch(y, device)
            if transpose:
                x, y = y, x
            qpe_val = qpe(flow, x, y)
            return to_numpy(qpe_val)

        return call_cv

    flow12, _ = train_flow(x1, x2)
    _, s12 = train_nqpe(x1, x2, flow12)
    flow21, _ = train_flow(x2, x1)
    _, s21 = train_nqpe(x2, x1, flow21)

    return wrapper(flow12), s12, wrapper(flow21, transpose=True), s21
