import numpy as np
import pandas as pd
import copy
import warnings

warnings.filterwarnings("ignore")

def log(this: pd.DataFrame) -> pd.DataFrame:
    outputToReturn = copy.copy(this)
    min_axis0 = np.nanmin(outputToReturn, axis=0, keepdims=True)
    outputToReturn = np.add(outputToReturn, np.abs(min_axis0)) + 1
    outputToReturn = np.log(outputToReturn)
    return outputToReturn

def abs_(this: pd.DataFrame) -> pd.DataFrame:
    outputToReturn = copy.copy(this)
    outputToReturn = np.abs(outputToReturn)
    return outputToReturn

def sign(this: pd.DataFrame) -> pd.DataFrame:
    outputToReturn = copy.copy(this)
    outputToReturn = np.sign(outputToReturn)
    return outputToReturn

# def exp(this: pd.DataFrame) -> pd.DataFrame:
#     outputToReturn = copy.copy(this)
#     outputToReturn = np.exp(outputToReturn)
#     return outputToReturn

# def rank(this: pd.DataFrame) -> pd.DataFrame:
#     outputToReturn = copy.copy(this)
#     outputToReturn = (np.argsort(np.argsort(outputToReturn)) + 1)\
#                                  /outputToReturn.shape[1]
#     return outputToReturn

# def scale(this: pd.DataFrame) -> pd.DataFrame:
#     outputToReturn = pd.DataFrame(copy.copy(this))
#     sum_axis1 = np.nansum(outputToReturn, axis=1, keepdims=True)
#     outputToReturn = np.divide(outputToReturn, sum_axis1)
#     return outputToReturn

def sigmoid(this: pd.DataFrame) -> pd.DataFrame:
    outputToReturn = copy.copy(this)
    sigmoid = 1 / (1 + np.exp(np.multiply(outputToReturn, -1)))
    outputToReturn = sigmoid
    return outputToReturn

######带符号的对数，𝑠𝑖𝑔𝑛(𝑎) ∗ 𝑙𝑜𝑔(𝑎𝑏𝑠(𝑎))#########
def s_log(this: pd.DataFrame) -> pd.DataFrame:
    outputToReturn = copy.copy(this)
    outputToReturn = np.sign(outputToReturn) * np.log(np.abs(outputToReturn)+1)
    return outputToReturn

######## 带符号的开方，𝑠𝑖𝑔𝑛(𝑎) ∗ 𝑠𝑞𝑟𝑡(𝑎𝑏𝑠(𝑎)) ###########
def s_sqrt(this: pd.DataFrame) -> pd.DataFrame:
    outputToReturn = copy.copy(this)
    outputToReturn = np.sign(outputToReturn) * np.sqrt(np.abs(outputToReturn))
    return outputToReturn


#
# data = pd.read_hdf('../data/T0-1/1325.h5')
# data['mid_price'] = (data['ask'] + data['bid'])/2
# for i in [1,10,20,100,200,400,600]:
#     data[f'mid_price_move_{i}'] = data['mid_price'].pct_change(i).shift(-i)
#     print(f'corr with {i} future days:',data[['average_bid_price',f'mid_price_move_{i}']].corr().iloc[0][1])
# X = data[['open', 'high', 'low', 'last_price', 'volume', 'turnover',
#        'trade_count', 'previous_close', 'high_limit', 'low_limit',
#        'total_volume', 'total_value', 'average_ask_price', 'average_bid_price']]
# Y = data['mid_price_move_10']
#
# log(X['open'])