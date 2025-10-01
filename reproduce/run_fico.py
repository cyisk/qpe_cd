import pandas as pd
from fico import *
from data.multivariate import *
from reproduce.multivariate import *
from reproduce.metric import *

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SEED = 0

if __name__ == '__main__':

    def FICO(x):
        return fico(x)

    results = pd.DataFrame(columns=[
        'dataset_id',
        'od',
        'odr',
        'delta_time',
    ])
    metrics = pd.DataFrame(columns=[
        'dataset_id',
        'od_mean',
        'od_std',
        'odr_mean',
        'odr_std',
        'dt_mean',
        'dt_std',
    ])

    for dataset_id in datasets:
        set_seed(seed=SEED)
        n_tests = datasets[dataset_id].n_datasets

        for pair_id in range(1, n_tests + 1):
            order, graph, delta_time = run_multivairate(
                dataset_id=dataset_id,
                pair_id=pair_id,
                method=FICO,
            )

            pair_od = od(order, graph)
            pair_odr = odr(order, graph)

            new_result = {
                'dataset_id': dataset_id,
                'od': pair_od,
                'odr': pair_odr,
                'delta_time': delta_time,
            }
            results.loc[len(results)] = new_result
            print(new_result)

        subdf = results[results['dataset_id'] == dataset_id]
        ods = subdf['od'].to_numpy()
        odrs = subdf['odr'].to_numpy()
        delta_times = subdf['delta_time'].to_numpy()

        new_metric = {
            'dataset_id': dataset_id,
            'od_mean': float(np.mean(ods)),
            'od_std': float(np.std(ods)),
            'odr_mean': float(np.mean(odrs)),
            'odr_std': float(np.std(odrs)),
            'dt_mean': float(np.mean(delta_times)),
            'dt_std': float(np.std(delta_times)),
        }
        print(new_metric)
        metrics.loc[len(metrics)] = new_metric

        results.to_csv(SCRIPT_DIR + '/fico_results.csv')
        metrics.to_csv(SCRIPT_DIR + '/fico_metrics.csv')
