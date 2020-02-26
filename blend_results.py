import optuna
import numpy as np
import pandas as pd
from functools import partial
from sklearn.metrics import mean_squared_error
from config import CONFIG as cfg


class GeneralizedMeanBlender():
    """Combines multiple predictions using generalized mean"""

    def __init__(self, p_range=(-2, 2)):
        """"""
        self.p_range = p_range
        self.p = None
        self.weights = None

    def _objective(self, trial, X, y):

        # create hyperparameters
        p = trial.suggest_uniform(f"p", *self.p_range)
        weights = [
            trial.suggest_uniform(f"w{i}", 0, 1)
            for i in range(X.shape[1])
        ]

        # blend predictions
        blend_preds, total_weight = 0, 0
        if p <= 0:
            for j, w in enumerate(weights):
                blend_preds += w * np.log1p(X[:, j])
                total_weight += w
            blend_preds = np.expm1(blend_preds / total_weight)
        else:
            for j, w in enumerate(weights):
                blend_preds += w * X[:, j] ** p
                total_weight += w
            blend_preds = (blend_preds / total_weight) ** (1 / p)

        # calculate mean squared error
        return np.sqrt(mean_squared_error(y, blend_preds))

    def fit(self, X, y, n_trials=10):
        # optimize objective
        obj = partial(self._objective, X=X, y=y)
        study = optuna.create_study()
        study.optimize(obj, n_trials=n_trials)
        # extract best weights
        if self.p is None:
            self.p = [v for k, v in study.best_params.items() if "p" in k][0]
        self.weights = np.array([v for k, v in study.best_params.items() if "w" in k])
        self.weights /= self.weights.sum()

    def transform(self, X):
        assert self.weights is not None and self.p is not None, \
            "Must call fit method before transform"
        if self.p == 0:
            return np.expm1(np.dot(np.log1p(X), self.weights))
        else:
            return np.dot(X ** self.p, self.weights) ** (1 / self.p)

    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)


def load_leak():
    leak = pd.read_feather(cfg.leak_df)
    leak.fillna(0, inplace=True)
    leak = leak[(leak.timestamp.dt.year > 2016) & (leak.timestamp.dt.year < 2019)]
    leak.loc[leak.meter_reading < 0, 'meter_reading'] = 0  # remove negative values
    leak = leak[leak.building_id != 245]
    return leak


if __name__ == '__main__':
    leak = load_leak()
    test = pd.read_feather(cfg.test_feather)
    for i, f in enumerate(cfg.submission_list):
        x = pd.read_csv(f, index_col=0).meter_reading
        x[x < 0] = 0
        test[f'pred{i}'] = x
    del x
    leak = pd.merge(leak, test[
        ['building_id', 'meter', 'timestamp', *[f"pred{i}" for i in range(len(cfg.submission_list))], 'row_id']], "left")
    # leak = pd.merge(leak, meta[['building_id', 'site_id']], 'left')

    for i in range(len(cfg.submission_list)):
        leak_score = np.sqrt(mean_squared_error(np.log1p(leak[f"pred{i}"]), np.log1p(leak.meter_reading)))
        print(f'submission {cfg.submission_list[i]} score{i}={leak_score}')
    X = np.log1p(leak[[f"pred{i}" for i in range(len(cfg.submission_list))]].values)
    y = np.log1p(leak["meter_reading"].values)
    gmb = GeneralizedMeanBlender()
    gmb.fit(X, y, n_trials=20)
    print("RMSE: {}".format(np.sqrt(mean_squared_error(gmb.transform(X), np.log1p(leak.meter_reading)))))
    print("-------------- Blend Results --------------")
    res = pd.read_csv(cfg.data_dir + "/sample_submission.csv")
    print("DEBUG: res shape {}".format(res.shape))
    X_test = test[[f"pred{i}" for i in range(len(cfg.submission_list))]].values
    res['meter_reading'] = np.expm1(gmb.transform(np.log1p(X_test)))
    res.loc[res.meter_reading < 0, 'meter_reading'] = 0
    # fill in leak data
    leak = leak[['meter_reading', 'row_id']].set_index('row_id').dropna()
    res.loc[leak.index, 'meter_reading'] = leak['meter_reading']
    print("DEBUG: res shape {}".format(res.shape))
    # save submission
    res.to_csv('submission_blend.csv', index=False)






