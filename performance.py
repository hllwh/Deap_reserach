import numpy as np
import pandas as pd
import empyrical


def annual_return(returns,fees):
    cum_return = (1 + returns).prod() - fees
    num_years = len(returns)/ (252*5400)
    return cum_return ** (1/num_years) - 1

def annual_volatility(returns):
    day_std = returns.std()
    return day_std * ((252*5400) ** 0.5)

def max_drawdown(returns):
    cum_return = np.cumprod(returns + 1)
    cum_return = cum_return.reset_index(drop=True)
    max_drawdown = 100
    for i in cum_return.index:
        if ((cum_return.iloc[i+1:]/cum_return.loc[i]).min() - 1 < max_drawdown):
            max_drawdown = (cum_return.iloc[i+1:]/cum_return.loc[i]).min() - 1
        # if ((cum_return.loc[i:].iloc[1:]/cum_return.loc[i]).min() - 1 < max_drawdown):
        #     max_drawdown = (cum_return.loc[i:].iloc[1:]/cum_return.loc[i]).min() - 1
    return max_drawdown

def calmar_ratio(returns,fees):
    annual_return_ = annual_return(returns,fees)
    max_drawdown_ = max_drawdown(returns)
    return annual_return_/max_drawdown_

def sharpe_ratio(returns,fees):
    risk_free = 0
    annual_return_ = annual_return(returns,fees)
    annual_volatility_ = annual_volatility(returns)
    return (annual_return_ - risk_free)/annual_volatility_

def win_rate(returns):
    return np.sum(returns>0)/(np.sum(returns>0) + np.sum(returns<0))

def performance(returns,dict,ID,fees):
    # annual_return = empyrical.annual_return(returns,period=periods)
    # annual_volatility = empyrical.annual_volatility(returns,period=periods)
    # max_drawdown = empyrical.max_drawdown(returns)
    # calmar = empyrical.calmar_ratio(returns,period=periods)
    # sharpe_ratio = empyrical.sharpe_ratio(returns,risk_free=0.00012,period=periods)
    # win_rate = np.sum(returns>0)/len(returns)
    annual_return_ = annual_return(returns,fees)
    annual_volatility_ = annual_volatility(returns)
    max_drawdown_ = max_drawdown(returns)
    calmar_ratio_ = abs(calmar_ratio(returns,fees))
    sharpe_ratio_ = sharpe_ratio(returns,fees)
    win_rate_ = win_rate(returns)
    # print('annual_return:',annual_return_)
    # print('annual_volatility:',annual_volatility_)
    # print('max_drawdown:', max_drawdown_)
    # print('calmar_ratio:', calmar_ratio_)
    # print('sharpe_ratio:', sharpe_ratio_)
    # print('win_rate', win_rate_)
    dict[ID] = {'annual_return':'{:.3f}'.format(annual_return_),
                'annual_volatility':'{:.3f}'.format(annual_volatility_),
                'max_drawdown':'{:.3f}'.format(max_drawdown_),
                'calmar_ratio':'{:.3f}'.format(calmar_ratio_),
                'sharpe_ratio':'{:.3f}'.format(sharpe_ratio_),
                'win_rate': '{:.3f}'.format(win_rate_),
                }

if __name__ == '__main__':
    data = pd.read_hdf('./data/T0-1/1325.h5')
    data['mid_price'] = (data['ask'] + data['bid']) / 2
    for i in [1, 10, 20, 100, 200, 400, 600]:
        data[f'mid_price_move_{i}'] = data['mid_price'].pct_change(i).shift(-i)
    test_data = data.iloc[int(0.8 * len(data)):]
    test_X = test_data[['open', 'high', 'low', 'last_price', 'volume', 'turnover',
                        'trade_count', 'previous_close', 'high_limit', 'low_limit',
                        'total_volume', 'total_value', 'average_ask_price', 'average_bid_price']]
    test_Y = test_data['mid_price_move_10']
    test_X_std = test_X.apply(lambda x: (x - np.mean(x)) / np.std(x))
    factor = pd.Series(list(test_data['mid_price_move_10'] + 1), index=test_data.index)
    #test_data['factor'] = list(test_data['open'] + 1)
    factor_data = pd.DataFrame(factor,columns=['factor',])
    factor_data['long_short'] = 0
    factor_data.loc[factor_data['factor'] > factor_data['factor'].quantile(0.9), 'long_short'] = 1
    factor_data.loc[factor_data['factor'] < factor_data['factor'].quantile(0.1), 'long_short'] = -1
    long_short = factor_data['long_short']
    # long_short = factor.apply(
    #     lambda x: 1 if x > factor.quantile(0.9) else (
    #         -1 if x < factor.quantile(0.1) else 0))
    returns = long_short * test_data['mid_price_move_1']
    performance(returns, dict, ID=111, fees=0)




    test_data = data.iloc[0:int(0.01 * len(data))]
    test_data['factor'] = list(test_data['open'] + 1)
    long_short= test_data['factor'].apply(
        lambda x: 1 if x > test_data['factor'].quantile(0.9) else (
            -1 if x < test_data['factor'].quantile(0.1) else 0))
    returns = long_short * test_data['mid_price_move_1']
    para_dict = dict()
    performance(returns,para_dict,ID = i+1,fees = 0)
    train_data = data.iloc[0:int(0.01 * len(data))]
    factor = 'mid_price_move_10'
    train_data['factor'] = train_data[factor].apply(
        lambda x: 1 if x > train_data[factor].quantile(0.9) else (
            -1 if x < train_data[factor].quantile(0.1) else 0))
    train_data['return'] = train_data['factor'] * train_data['mid_price_move_1']
    para_dict = dict()
    performance(train_data['return'], para_dict, factor, fees = 0)
    print('done')

