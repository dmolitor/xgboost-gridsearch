import pandas as pd
from tqdm import tqdm
from utils import expand_grid, validate_kwargs
import xgboost as xgb

class GridSearch:
    """A simple class to perform parameter grid search using xgboost.cv"""
    def __init__(self, param_grid: dict, **kwargs):
        """Key word arguments are passed directly in to xgboost.cv"""
        validate_kwargs(**kwargs)
        self._booster_args = kwargs
        if "metrics" in kwargs:
            metrics = kwargs.get("metrics")
            if isinstance(metrics, list):
                target_metric = metrics[len(metrics) - 1]
            else:
                target_metric = metrics
            self._target_metric = target_metric
        self.best_idx = None
        self.best_model = None
        self.best_parameters = None
        self.param_grid = expand_grid(param_grid)
    
    def _best_parameters(self, min: bool = True):
        """
        Find the optimal model paramters based on minimized (or maximized)
        cross-validation evaluation metrics
        """
        target_metric = self._target_metric
        if target_metric is None:
            target_metrics = self.model_metrics.iloc[:, 2]
        else:
            target_metrics = self.model_metrics[f"test-{target_metric}-mean"]
        if min:
            optimal_idx = target_metrics.idxmin()
        else:
            optimal_idx = target_metrics.idxmax()
        optimal_params = (
            self
            .param_grid
            .iloc[optimal_idx]
            .to_dict()
        )
        optimal_num_boost = (
            self
            .model_metrics
            .iloc[optimal_idx]["num_boost_round"]
        )
        self.best_idx = optimal_idx
        self.best_parameters = optimal_params
        self.best_num_boost_round = int(optimal_num_boost)
    
    def _drop_cv_args(self, **kwargs):
        train_args = [
            "params",
            "dtrain",
            "num_boost_round",
            "evals",
            "obj",
            "feval",
            "maximize",
            "evals_result",
            "verbose_eval",
            "xgb_model",
            "callbacks",
            "custom_metric"
        ]
        for k in list(kwargs.keys()).copy():
            if k not in train_args:
                del kwargs[k]
        return kwargs
    
    def _fit_with_params(self, params: dict) -> pd.DataFrame:
        """Fit a boosted model with a single set of parameter values"""
        model = xgb.cv(params=params, **self._booster_args)
        model_error = model.iloc[(len(model) - 1):len(model)]
        model_error = model_error.assign(num_boost_round = len(model))
        return model_error
    
    def _fit_param_grid(self, verbose: bool = True) -> pd.DataFrame:
        """Fit a boosted model across every row in our parameter grid"""
        param_grid = self.param_grid
        model_error = []
        if verbose:
            param_iter = tqdm(param_grid.iterrows(), total=len(param_grid))
        else:
            param_iter = param_grid.iterrows()
        for params in param_iter:
            error = self._fit_with_params(params=params[1].to_dict())
            model_error.append(error)
        model_error_df = pd.concat(model_error, axis=0).reset_index(drop=True)
        self.model_metrics = model_error_df
    
    def fit(self,
            verbose: bool = True,
            minimize_cv_metric: bool = True,
            refit: bool = True):
        """
        Fit a boosted tree model across every parameter in our tuning grid.
        Then, find the optimal model parameters and, if desired, fit a final
        boosted tree model using these optimal parameters.

        Parameters
        ----------
        verbose:
            A boolean. Display model training progress across tuning grid.
        minimize_cv_metric:
            A boolean. True indicates the optimal parameters are those that
            minimize the cross-validation performance metric. Otherwise the
            optimal parameters are those that maximize the performance metric.
        refit:
            A boolean. Refit a boosted tree using the optimal parameters on
            the whole dataset.
        """
        self._fit_param_grid(verbose=verbose)
        self._best_parameters(min=minimize_cv_metric)
        if refit:
            optimal_params = self.best_parameters
            booster_args = self._drop_cv_args(**self._booster_args)
            booster_args["num_boost_round"] = self.best_num_boost_round
            best_model = xgb.train(params=optimal_params, **booster_args)
            self.best_model = best_model
