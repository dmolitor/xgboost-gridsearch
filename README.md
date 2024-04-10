# XGBoost Gridsearch

This module allows you to combine grid-search tuning with early stopping via 
cross validation.

## Dependencies
This requires the following dependencies:
```python
pip install httpimport pandas tqdm xgboost
```

## Importing

You can import this code from Github via the `httpimport` module as follows:
```python
import httpimport as hi

with hi.github_repo("dmolitor", "xgboost-gridsearch", ref="main"):
    from xgb_gridsearch.grid_search import GridSearch
```

## Predicting proportion of working mothers with 1-year-olds

Two important things to note. 

- The `GridSearch` class takes as its first argument `param_grid` which is
 the same as in `sklearn.model_selection.GridSearchCV`.
- All other keyword arguments are passed directly to `xgboost.cv`.

Here is a list of all
[parameter options](https://xgboost.readthedocs.io/en/stable/parameter.html),
and here is the
[documentation](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.cv)
for `xgboost.cv`.

```python
import httpimport as hi
import json
import pandas as pd
import xgboost as xgb

with hi.github_repo("dmolitor", "xgboost-gridsearch", ref="main"):
    from xgb_gridsearch.grid_search import GridSearch

# Import data
parents = pd.read_csv("https://raw.githubusercontent.com/info3370/info3370.github.io/main/data/parents.csv")
parents["sex"] = parents.sex.astype("category")

# Make a testing set
parents_test = pd.DataFrame([{"sex": "female", "child_age": 1}])
parents_test["sex"] = parents_test.sex.astype("category")

# Get into DMatrix form
data_train = xgb.DMatrix(
    data = parents.drop("at_work", axis=1),
    label=parents[["at_work"]],
    enable_categorical=True
)
data_test = xgb.DMatrix(data = parents_test, enable_categorical=True)

# Set parameters to grid search across
params = {
    "objective": ["binary:logistic"],
    "learning_rate": [0.1, 0.3],
    "max_depth": [4, 6, 8],
    "grow_policy": ["depthwise"],
    "min_child_weight": [0.5, 1, 3],
    "max_leaves": [0, 3, 5, 7]
}

# Create grid search object
grid_search = GridSearch(
    param_grid=params,
    dtrain=data_train,
    num_boost_round=1000,
    nfold=5,
    shuffle=True,
    metrics=["auc", "logloss"],
    early_stopping_rounds=5,
    verbose_eval=False
)
grid_search.fit(verbose=True, minimize_cv_metric=True, refit=True)

# Get predicted proportion of working mothers with 1-year-olds
prop_working = round(grid_search.best_model.predict(data_test)[0], 4)

# Print optimal parameters
print(json.dumps(grid_search.best_parameters, indent=4))
print("Predicted proportion working mothers:", prop_working)
```
