import time
import random
import numpy as np
import torch as th
from sklearn.preprocessing import StandardScaler
from typing import Callable, Tuple

from data.multivariate import *

datasets = {
    'anm-sf-5': Simulated_ANM_SF_5,
    'anm-sf-10': Simulated_ANM_SF_10,
    'anm-sf-20': Simulated_ANM_SF_20,
    'anm-sf-50': Simulated_ANM_SF_50,
    'anm-sf-100': Simulated_ANM_SF_100,
    'anm-er-5': Simulated_ANM_ER_5,
    'anm-er-10': Simulated_ANM_ER_10,
    'anm-er-20': Simulated_ANM_ER_20,
    'anm-er-50': Simulated_ANM_ER_50,
    'anm-er-100': Simulated_ANM_ER_100,
    'lsnm-sf-5': Simulated_LSNM_SF_5,
    'lsnm-sf-10': Simulated_LSNM_SF_10,
    'lsnm-sf-20': Simulated_LSNM_SF_20,
    'lsnm-sf-50': Simulated_LSNM_SF_50,
    'lsnm-sf-100': Simulated_LSNM_SF_100,
    'lsnm-er-5': Simulated_LSNM_ER_5,
    'lsnm-er-10': Simulated_LSNM_ER_10,
    'lsnm-er-20': Simulated_LSNM_ER_20,
    'lsnm-er-50': Simulated_LSNM_ER_50,
    'lsnm-er-100': Simulated_LSNM_ER_100,
    'anm-c-sf-5': Simulated_ANM_C_SF_5,
    'anm-c-sf-10': Simulated_ANM_C_SF_10,
    'anm-c-sf-20': Simulated_ANM_C_SF_20,
    'anm-c-sf-50': Simulated_ANM_C_SF_50,
    'anm-c-sf-100': Simulated_ANM_C_SF_100,
    'anm-c-er-5': Simulated_ANM_C_ER_5,
    'anm-c-er-10': Simulated_ANM_C_ER_10,
    'anm-c-er-20': Simulated_ANM_C_ER_20,
    'anm-c-er-50': Simulated_ANM_C_ER_50,
    'anm-c-er-100': Simulated_ANM_C_ER_100,
    'lsnm-c-sf-5': Simulated_LSNM_C_SF_5,
    'lsnm-c-sf-10': Simulated_LSNM_C_SF_10,
    'lsnm-c-sf-20': Simulated_LSNM_C_SF_20,
    'lsnm-c-sf-50': Simulated_LSNM_C_SF_50,
    'lsnm-c-sf-100': Simulated_LSNM_C_SF_100,
    'lsnm-c-er-5': Simulated_LSNM_C_ER_5,
    'lsnm-c-er-10': Simulated_LSNM_C_ER_10,
    'lsnm-c-er-20': Simulated_LSNM_C_ER_20,
    'lsnm-c-er-50': Simulated_LSNM_C_ER_50,
    'lsnm-c-er-100': Simulated_LSNM_C_ER_100,
    'gc-sf-5': Simulated_GC_SF_5,
    'gc-sf-10': Simulated_GC_SF_10,
    'gc-sf-20': Simulated_GC_SF_20,
    'gc-sf-50': Simulated_GC_SF_50,
    'gc-sf-100': Simulated_GC_SF_100,
    'gc-er-5': Simulated_GC_ER_5,
    'gc-er-10': Simulated_GC_ER_10,
    'gc-er-20': Simulated_GC_ER_20,
    'gc-er-50': Simulated_GC_ER_50,
    'gc-er-100': Simulated_GC_ER_100,
    'lingam-sf-5': Simulated_LINGAM_SF_5,
    'lingam-sf-10': Simulated_LINGAM_SF_10,
    'lingam-sf-20': Simulated_LINGAM_SF_20,
    'lingam-sf-50': Simulated_LINGAM_SF_50,
    'lingam-sf-100': Simulated_LINGAM_SF_100,
    'lingam-er-5': Simulated_LINGAM_ER_5,
    'lingam-er-10': Simulated_LINGAM_ER_10,
    'lingam-er-20': Simulated_LINGAM_ER_20,
    'lingam-er-50': Simulated_LINGAM_ER_50,
    'lingam-er-100': Simulated_LINGAM_ER_100,
    'lsnm-sf-100[100]': Simulated_LSNM_SF_100_N100,
    'lsnm-sf-100[200]': Simulated_LSNM_SF_100_N200,
    'lsnm-sf-100[500]': Simulated_LSNM_SF_100_N500,
    'lsnm-sf-100[800]': Simulated_LSNM_SF_100_N800,
    'lsnm-sf-100[1000]': Simulated_LSNM_SF_100_N1000,
    'lsnm-sf-100[2000]': Simulated_LSNM_SF_100_N2000,
    'lsnm-sf-100[3000]': Simulated_LSNM_SF_100_N3000,
    'lsnm-er-100[100]': Simulated_LSNM_ER_100_N100,
    'lsnm-er-100[200]': Simulated_LSNM_ER_100_N200,
    'lsnm-er-100[500]': Simulated_LSNM_ER_100_N500,
    'lsnm-er-100[800]': Simulated_LSNM_ER_100_N800,
    'lsnm-er-100[1000]': Simulated_LSNM_ER_100_N1000,
    'lsnm-er-100[2000]': Simulated_LSNM_ER_100_N2000,
    'lsnm-er-100[3000]': Simulated_LSNM_ER_100_N3000,
    'sachs': Sachs,
    'syntren': Syntren,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def run_multivairate(
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
    x = data.data
    graph = data.graph.detach().cpu().numpy()

    if shuffle:
        idx = np.random.permutation(x.shape[1])
        x = x[:, idx]
        graph = graph[idx, :][:, idx]

    start_time = time.time()
    order = method(x)
    end_time = time.time()
    delta_time = end_time - start_time

    return order, graph, delta_time
