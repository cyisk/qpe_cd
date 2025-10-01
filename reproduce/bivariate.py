import time
import random
import numpy as np
import torch as th
from sklearn.preprocessing import StandardScaler
from typing import Callable, Tuple

from data.bivariate import *

datasets = {
    'an': AN,
    'ans': ANs,
    'ls': LS,
    'lss': LSs,
    'mnu': MNU,
    'sim': SIM,
    'simc': SIMc,
    'simg': SIMG,
    'simln': SIMln,
    'tue': Tuebingen,
    'cha': Cha,
    'net': Net,
    'multi': Multi,
    'd4s1': D4S1,
    'd4s2a': D4S2A,
    'd4s2b': D4S2B,
    'd4s2c': D4S2C,
    'per': PER,
    'sig': SIG,
    'vex': VEX,
    'qdv': QDV,
    'sigv': SIGV,
    'rbfv': RBFV,
    'nnv': NNV,
}

excludings = {
    'tue': [47, 52, 53, 54, 55, 70, 71, 105, 107],
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def run_bivairate(
    dataset_id: str,
    pair_id: int,
    method: Callable,
    shuffle: bool = True,
) -> Tuple[int, float] | None:
    data = datasets[dataset_id](
        pair_id=pair_id,
        double=True,
        preprocessor=StandardScaler(),
    )
    if dataset_id in excludings and pair_id in excludings[dataset_id]:
        return None

    cause = data.cause[:, :1].cpu().numpy()
    effect = data.effect[:, :1].cpu().numpy()
    start_time = time.time()
    if shuffle:
        if random.random() > 0.5:
            direction = method(cause, effect)
        else:
            direction = -method(effect, cause)
    else:
        direction = method(cause, effect)
    end_time = time.time()
    delta_time = end_time - start_time

    return direction, delta_time
