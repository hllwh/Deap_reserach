import numpy as np
import copy
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


def add_(this: pd.DataFrame, that: pd.DataFrame) -> pd.DataFrame:
    assert this.shape == that.shape

    outputToReturn = copy.copy(this)
    outputToReturn = np.add(this, that)
    return outputToReturn

def subtract_(this: pd.DataFrame, that: pd.DataFrame) -> pd.DataFrame:
    assert this.shape == that.shape

    outputToReturn = copy.copy(this)
    outputToReturn= np.subtract(this, that)
    return outputToReturn

def multiply_(this: pd.DataFrame, that: pd.DataFrame) -> pd.DataFrame:
    assert this.shape == that.shape

    outputToReturn = copy.copy(this)
    outputToRetur = np.multiply(this, that)
    return outputToReturn

def divide_(this: pd.DataFrame, that: pd.DataFrame) -> pd.DataFrame:
    this = pd.DataFrame(this)
    that = pd.DataFrame(that)
    assert this.shape == that.shape
    nrows, ncols = this.shape
    nan2DArray = np.full((nrows, ncols), np.nan)
    outputToReturn = copy.copy(this)
    outputToReturn = np.divide(this, that,
                             out = np.full_like(nan2DArray, 10000),where = that!=0)
    return outputToReturn

