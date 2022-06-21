import numpy as np
import copy
import pandas as pd
import utils
import bottleneck as bn
import warnings

warnings.filterwarnings("ignore")


def ts_sum(df:pd.Series,
           window=10)->pd.Series:
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas Series.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """

    return df.rolling(window, min_periods=window//2).sum()


# def sma_bn(df:pd.DataFrame,
#            window:int=10)->pd.DataFrame:
#     """
#     此方程與ts_mean非常相像，建議將此方程刪除並將ts_mean改為此方程之方法
#     Wrapper function to estimate SMA.
#     :param df: a pandas DataFrame.
#     :param window: the rolling window.
#     :return: a pandas DataFrame with the time-series min over the past 'window' days.
#     """
#     if window == 1:
#         window = 2
#     return pd.DataFrame(bn.move_mean(df, window=window, min_count=window//2,axis=0),columns = df.columns,index = df.index)

def ts_mean_bn(df:pd.DataFrame,
               window:int=10)-> pd.DataFrame:
    """
    此方程與sma非常相像，建議將此方程改為ts_mean之方法，並將sma刪除
    Wrapper function to estimate ts_mean.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    if window == 1:
        window = 2
    return pd.Series(bn.move_mean(df, window=window,axis=0),index = df.index)
#
def ts_std_bn(df:pd.DataFrame,
              window:int=10)-> pd.DataFrame:
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    if window == 1:
        window = 2
    return pd.Series(bn.move_std(df, window=window, min_count=window//2,axis=0),index = df.index)

def ts_max_bn(df:pd.DataFrame,
              window:int=10)-> pd.DataFrame:
    """
    Wrapper function to estimate rolling max.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    if window == 1:
        window = 2
    return pd.Series(bn.move_max(df, window=window, min_count=window//2,axis=0),index = df.index)

def ts_min_bn(df:pd.DataFrame,
              window:int=10)-> pd.DataFrame:
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    if window == 1:
        window = 2
    return pd.Series(bn.move_min(df, window=window, min_count=window//2,axis=0),index = df.index)

# def ts_argmax_bn(df:pd.DataFrame,
#                  window:int=10)-> pd.DataFrame:
#     """
#     Wrapper function to estimate which day ts_max(df, window) occurred on
#     :param df: a pandas DataFrame.
#     :param window: the rolling window.
#     :return: well.. that :)
#     """
#     if window == 1:
#         window = 2
#     deduction = np.array([range(1,df.shape[0]+1)]).T
#     deduction[deduction > window] = window
#     return pd.DataFrame(deduction - bn.move_argmax(df, window=window,min_count=window//2,axis=0),columns = df.columns,index = df.index)
#
# def ts_argmin_bn(df:pd.DataFrame,
#                  window:int=10)-> pd.DataFrame:
#     """
#     Wrapper function to estimate which day ts_min(df, window) occurred on
#     :param df: a pandas DataFrame.
#     :param window: the rolling window.
#     :return: well.. that :)
#     """
#     if window == 1:
#         window = 2
#     deduction = np.array([range(1,df.shape[0]+1)]).T
#     deduction[deduction > window] = window
#     return pd.DataFrame(deduction - bn.move_argmin(df, window=window,min_count=window//2,axis=0),columns = df.columns,index = df.index)

# def stddev(df:pd.Series,
#            window=10)->pd.Series:
#     """
#     Wrapper function to estimate rolling standard deviation.
#     :param df: a pandas DataFrame.
#     :param window: the rolling window.
#     :return: a pandas DataFrame with the time-series min over the past 'window' days.
#     """
#     return df.rolling(window, min_periods=window//2).std()
#
# def avg(df:pd.Series,
#            window=10)->pd.Series:
#     """
#     Wrapper function to estimate rolling mean.
#     :param df: a pandas DataFrame.
#     :param window: the rolling window.
#     :return: a pandas DataFrame with the time-series min over the past 'window' days.
#     """
#     return df.rolling(window, min_periods=window//2).mean()
#
#
# def ts_rank(df:pd.Series,
#             window=10)-> pd.Series:
#     return pd.concat([df.iloc[_-window:_].rank(pct=True).iloc[-1] for _ in range(window, len(df)+1)], axis=1).T.reindex(df.index)
#
#
#
#
# def ts_min(df:pd.Series,
#             window=10)-> pd.Series:
#     """
#     Wrapper function to estimate rolling min.
#     :param df: a pandas DataFrame.
#     :param window: the rolling window.
#     :return: a pandas DataFrame with the time-series min over the past 'window' days.
#     """
#     return df.rolling(window, min_periods=window//2).min()
#
#
# def ts_max(df:pd.Series,
#             window=10)-> pd.Series:
#     """
#     Wrapper function to estimate rolling min.
#     :param df: a pandas DataFrame.
#     :param window: the rolling window.
#     :return: a pandas DataFrame with the time-series max over the past 'window' days.
#     """
#     return df.rolling(window, min_periods=window//2).max()


def delta(df:pd.Series,
          period = 1)-> pd.Series:
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.diff(period)


def delay(df:pd.Series,
          period = 1)-> pd.Series:
    """
    Wrapper function to estimate lag.
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.shift(period)


# def rank(df:pd.Series)-> pd.Series:
#     """
#     Cross sectional rank
#     :param df: a pandas DataFrame.
#     :return: a pandas DataFrame with rank along columns.
#     """
#     return df.rank(axis=1, pct=True)


# def scale(df:pd.Series,
#           k=1)-> pd.Series:
#     """
#     Scaling time serie.
#     :param df: a pandas DataFrame.
#     :param k: scaling factor.
#     :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
#     """
#     return df.mul(k).div(np.abs(df).sum())


# def ts_argmax(df:pd.Series,
#               window=10)-> pd.Series:
#     """
#     Wrapper function to estimate which day ts_max(df, window) occurred on
#     :param df: a pandas DataFrame.
#     :param window: the rolling window.
#     :return: well.. that :)
#     """
#     return df.rolling(window, min_periods=window//2).apply(np.argmax) + 1
#
#
# def ts_argmin(df:pd.Series,
#               window=10)-> pd.Series:
#     """
#     Wrapper function to estimate which day ts_min(df, window) occurred on
#     :param df: a pandas DataFrame.
#     :param window: the rolling window.
#     :return: well.. that :)
#     """
#     return df.rolling(window, min_periods=window//2).apply(np.argmin) + 1


# def delay(this: pd.DataFrame, aNum: int = 1) -> pd.DataFrame:
#     assert aNum >= 0
#     outputToReturn = copy.copy(this)
#     tmp_copy = outputToReturn.copy()
#
#     # tmp_copy[-aNum:, :] = np.nan
#     roll = np.roll(tmp_copy, aNum, axis=0)
#     roll[: aNum, :] = tmp_copy[:aNum, :]
#     outputToReturn = roll
#     return outputToReturn
#
# # 𝑑𝑒𝑙𝑡𝑎(𝑎, 𝑏) 𝑎 − 𝑑𝑒𝑙𝑎𝑦(𝑎, 𝑏)
# def delta(this: pd.DataFrame, aNum: int = 1) -> pd.DataFrame:
#     assert aNum >= 0
#     outputToReturn = copy.copy(this)
#     tmp_copy = outputToReturn.copy()
#
#     roll = np.roll(tmp_copy, aNum, axis=0)
#     roll[: aNum, :] = tmp_copy[:aNum, :]
#
#     outputToReturn.generalData = np.subtract(tmp_copy, roll)
#     return outputToReturn
#
#
# # 𝑑𝑒𝑙𝑡𝑎(𝑎, 𝑏) 𝑎 − 𝑑𝑒𝑙𝑎𝑦(𝑎, 𝑏)
# def pct_change(this: pd.DataFrame, aNum: int = 1) -> pd.DataFrame:
#     assert aNum >= 0
#     outputToReturn = copy.copy(this)
#     tmp_copy = outputToReturn.copy()
#
#     roll = np.roll(tmp_copy, aNum, axis=0)
#     roll[: aNum,:] = tmp_copy[:aNum, :]
#
#     delay = roll
#     delta = np.subtract(tmp_copy, roll)
#
#     outputToReturn = np.divide(delta, delay)
#     return outputToReturn
#
#
#
# def ts_max(this: pd.DataFrame, rollingDaysN: int = 2) -> pd.DataFrame:
#     assert rollingDaysN >= 0
#
#     outputToReturn = copy.copy(this)
#     toStride2DArray = pd.DataFrame(outputToReturn)
#     strided = utils.get_strided(toStride2DArray, rollingDaysN)
#     max_ = np.nanmax(strided, axis=1)
#     outputToReturn = max_
#     return outputToReturn
#
#
# def ts_min(this: pd.DataFrame, rollingDaysN: int = 2) -> pd.DataFrame:
#     assert rollingDaysN >= 0
#     outputToReturn = copy.copy(this)
#     toStride2DArray = pd.DataFrame(outputToReturn)
#     strided = utils.get_strided(toStride2DArray, rollingDaysN)
#     min_ = np.nanmin(strided, axis=1)
#     outputToReturn = min_
#     return outputToReturn
#
# def ts_decay_linear(this: pd.DataFrame, rollingDaysN: int = 2) -> pd.DataFrame:
#     assert rollingDaysN >= 0
#     outputToReturn = copy.copy(this)
#     toStride2DArray = pd.DataFrame(outputToReturn)
#     strided = utils.get_strided(toStride2DArray, rollingDaysN)
#     todivide = (1+rollingDaysN)*rollingDaysN/2
#     weight = np.array(range(1,rollingDaysN+1))/todivide
#     toReturn = np.average(strided, axis=1,weights= weight)
#     outputToReturn = toReturn
#     return outputToReturn
#
# def ts_argmax(this: pd.DataFrame, rollingDaysN: int = 2) -> pd.DataFrame:
#     assert rollingDaysN >= 0
#     outputToReturn = copy.copy(this)
#     toStride2DArray = pd.DataFrame(outputToReturn)
#     strided = utils.get_strided(toStride2DArray, rollingDaysN)
#     temparray = np.zeros((strided.shape[0], strided.shape[2]))
#     for i in range(strided.shape[0]):
#         tempArg = np.argsort(strided[i],axis=0)
#         argmax = tempArg[-1,:]
#         temparray[i] = argmax
#     outputToReturn = temparray
#     return outputToReturn
#
# def ts_argmin(this: pd.DataFrame, rollingDaysN: int = 2) -> pd.DataFrame:
#     assert rollingDaysN >= 0
#     outputToReturn = copy.copy(this)
#     toStride2DArray = pd.DataFrame(outputToReturn)
#     strided = utils.get_strided(toStride2DArray, rollingDaysN)
#     temparray = np.zeros((strided.shape[0], strided.shape[2]))
#     for i in range(strided.shape[0]):
#         tempArg = np.argsort(strided[i],axis=0)
#         argmax = tempArg[0,:]
#         temparray[i] = argmax
#     outputToReturn = temparray
#     return outputToReturn