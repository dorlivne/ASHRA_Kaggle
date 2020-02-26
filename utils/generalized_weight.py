import optuna
import numpy as np
from functools import partial
from sklearn.metrics import mean_squared_error


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
