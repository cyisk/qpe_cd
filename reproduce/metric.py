import numpy as np

from numpy.typing import NDArray


def prec(flags: NDArray) -> float:
    flags = flags.copy()
    flags[~(flags == 1)] = 0
    return float(np.mean(flags))


def audrc(flags: NDArray) -> float:
    flags = flags.copy()
    flags[~(flags == 1)] = 0
    return float(np.mean(np.cumsum(flags) / (np.arange(flags.size) + 1)))


def od(order: NDArray, graph: NDArray) -> float:
    err = 0
    for i in range(len(order)):
        err += graph[order[i + 1:], order[i]].sum()
    return float(err)


def odr(order: NDArray, graph: NDArray) -> float:
    return float(od(order, graph) / np.sum(graph))
