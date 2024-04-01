from .exceptions import ArgumentValueError, ParameterFormatError
from itertools import product
import pandas as pd
from typing import Any

def expand_grid(params: dict) -> pd.DataFrame:
    """Replicates R's expand.grid function"""
    validate_params(params)
    cols = list(params.keys())
    grid = list(product(*params.values()))
    grid = pd.DataFrame(grid, columns=cols)
    return grid

def is_nonempty_list(x: list | Any) -> bool:
    is_list = isinstance(x, list)
    empty = not x
    valid = is_list and not empty
    return valid

def validate_kwargs(**kwargs) -> bool:
    """Throws an error if any kwargs are None or an empty list"""
    for k, v in kwargs.items():
        valid = v is not None
        if valid and isinstance(v, list):
            valid = is_nonempty_list(v)
        if not valid:
            raise ArgumentValueError(
                f"Argument `{k}` must not be None if provided"
            )
    return True

def validate_params(params: dict) -> bool:
    """Throws an error if any parameters are unboxed (aka not a list)"""
    for k, v in params.items():
        valid = is_nonempty_list(v)
        if not valid:
            raise ParameterFormatError(
                f"Parameter `{k}` must be a non-empty list"
            )
    return True
