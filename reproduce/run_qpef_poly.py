import pandas as pd
from qpef import *
from data.bivariate import *
from reproduce.bivariate import *
from reproduce.metric import *

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SEED = 0
DEGREE = 0

if __name__ == '__main__':

    def QPE_f(x1, x2):
        qpe_12, score_12, qpe_21, score_21 = qpe_f(
            x1,
            x2,
            transform_cls=UMNNConditionalTransform,
            transforms=1,
            neural_qpe_cls=PolynomialBasisNeuralQPE,
            neural_qpe_kwargs={
                'degree': DEGREE,
            },
            epochs=1000,
            learning_rate=0.01,
            weight_decay=0.001,
            hidden_features=(100, 100),
            activation=th.nn.SiLU,
        )
        return 1 if score_12 > score_21 else -1

    results = pd.DataFrame(columns=[
        'dataset_id',
        'pair_id',
        'direction',
        'delta_time',
    ])
    metrics = pd.DataFrame(columns=[
        'dataset_id',
        'prec',
        'audrc',
        'dt_mean',
        'dt_std',
    ])

    for dataset_id in datasets:
        set_seed(seed=SEED)
        n_tests = datasets[dataset_id].n_datasets

        for pair_id in range(1, n_tests + 1):
            pair_result = run_bivairate(
                dataset_id=dataset_id,
                pair_id=pair_id,
                method=QPE_f,
            )
            if pair_result is None:
                continue
            direction, delta_time = pair_result

            new_result = {
                'dataset_id': dataset_id,
                'pair_id': pair_id,
                'direction': direction,
                'delta_time': delta_time,
            }
            results.loc[len(results)] = new_result

        subdf = results[results['dataset_id'] == dataset_id]
        directions = subdf['direction'].to_numpy()
        delta_times = subdf['delta_time'].to_numpy()

        new_metric = {
            'dataset_id': dataset_id,
            'prec': prec(directions),
            'audrc': audrc(directions),
            'dt_mean': float(np.mean(delta_times)),
            'dt_std': float(np.std(delta_times)),
        }
        print(new_metric)
        metrics.loc[len(metrics)] = new_metric

        results.to_csv(SCRIPT_DIR + '/qpef_results_poly.csv')
        metrics.to_csv(SCRIPT_DIR + '/qpef_metrics_poly.csv')
