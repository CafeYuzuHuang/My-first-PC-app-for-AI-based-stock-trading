# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from myUtilities import myParams

"""
myAna.py 所具有的功能
(1) 財務與風險指標評估
(2) 訓練結果視覺化
備註：財務與風險指標評估第(5)大項中，未平倉的交易不會被計入
"""

# Global variables and constants:
plt.set_loglevel("warning") # avoid log polluted by mpl

# --- --- #

def Training_Visualizer(metrics_df, isembedded = False):
    """
    Training analytics: 
    Arg.: metrics_df with columns
        - loss
        - val_loss
        - reward
        - epsilon
          isembedded: embedded to GUI (e.g. tk widget) or not
    """
    plt.style.use('dark_background')
    mpl.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "lines.linewidth": 1,
            "lines.markersize": 10
        }
    )
    
    x = metrics_df.index.values
    y1 = metrics_df.loss.values.reshape(x.shape)
    y2 = metrics_df.val_loss.values.reshape(x.shape)
    y3 = metrics_df.reward.values.reshape(x.shape)
    y4 = metrics_df.epsilon.values.reshape(x.shape)
    y_zero = pd.DataFrame(data = np.zeros(len(y3)))
    y_zero = y_zero.values.reshape(x.shape)
    
    grid_kws = {"height_ratios": (.25, .25, .25, .25), "hspace": .1}
    fig, axs = plt.subplots(4, sharex = True, squeeze = True, \
                            figsize = (12, 8), gridspec_kw = grid_kws)
    fig.suptitle("Training visualization")
    axs[0].plot(x, y1, color = 'c')
    axs[1].plot(x, y2, color = 'y')
    axs[2].plot(x, y3, color = 'm')
    axs[2].plot(x, y_zero, color = 'm')
    axs[2].fill_between(x, y_zero, y3, color = 'm', alpha = 0.35)
    axs[3].plot(x, y4, color = 'b')
    axs[0].set_ylabel("loss")
    axs[1].set_ylabel("val_loss")
    axs[2].set_ylabel("reward")
    axs[3].set_ylabel("epsilon")
    axs[3].set_xlabel("Epochs")
    axs[3].set_xlim([min(x), max(x)])
    axs[3].set_ylim([0, 1])
    axs[3].xaxis.set_major_locator(MaxNLocator(integer = True))
    if isembedded:
        return fig, axs
    else:
        plt.show()
        return None

def Asset_Visualizer(asset_record_df, isembedded = False):
    """
    Trade performance - asset vs. iteration visualization
    Arg.: asset_record_df with columns
        - current iteration
        - previous avg. stock price
        - current asset = cash + stock value
        - current cash
        - current stock value
        - current action
        - current position (after action)
          isembedded: embedded to GUI (e.g. tk widget) or not
    """
    plt.style.use('dark_background')
    mpl.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "lines.linewidth": 1,
            "lines.markersize": 10
        }
    )
    
    x = asset_record_df.iteration.values[1:]
    xs = x.shape
    y1 = asset_record_df.avg_price.values[1:].reshape(xs)
    y21 = asset_record_df.cash_and_stock.values[1:].reshape(xs)
    y22 = asset_record_df.cash.values[1:].reshape(xs)
    y23 = asset_record_df.stock_value.values[1:].reshape(xs)
    y31 = asset_record_df.action.values[1:].reshape(xs)
    y32 = asset_record_df.position.values[1:].reshape(xs)
    y_zero = pd.DataFrame(data = np.zeros(len(y32)))
    y_zero = y_zero.values.reshape(xs)
    ic0 = asset_record_df.cash_and_stock.values[0]
    n = len(asset_record_df.cash_and_stock.values[1:])
    y_ic = pd.DataFrame(data = np.array([ic0] * n))
    y_ic = y_ic.values.reshape(xs)
    
    grid_kws = {"height_ratios": (.3, .3, .3), "hspace": .1}
    fig, axs = plt.subplots(3, sharex = True, squeeze = True, \
                            figsize = (12, 8), gridspec_kw = grid_kws)
    fig.suptitle("Trading performance visualization")
    axs[0].plot(x, y1, color = 'b')
    axs[0].set_ylabel("Previous mid price (NT$)")
    axs[1].plot(x, y21, color = 'y', label = "asset")
    axs[1].fill_between(x, y_ic, y21, color = 'y', alpha = 0.35)
    axs[1].plot(x, y22, color = 'c', label = "cash")
    axs[1].plot(x, y23, color = 'm', label = "security")
    axs[1].legend(loc = "best", fancybox = True, framealpha = 0.5)
    axs[1].set_ylabel("Values (NT$)")
    axs[2].plot(x, y31, color = 'g', label = "action")
    axs[2].plot(x, y32, color = 'r', label = "position")
    axs[2].fill_between(x, y_zero, y32, color = 'r', alpha = 0.35)
    axs[2].legend(loc = "best", fancybox = True, framealpha = 0.5)
    axs[2].set_ylim([-1, 1])
    axs[2].set_ylabel("Trade signal")
    axs[2].set_xlim([min(x), max(x)])
    axs[2].xaxis.set_major_locator(MaxNLocator(integer = True))
    axs[2].set_xlabel("Iterations")
    axs[0].grid(which = "major", axis = "both", linestyle = '-', \
                color = "gray", linewidth = 0.5)
    axs[1].grid(which = "major", axis = "both", linestyle = '-', \
                color = "gray", linewidth = 0.5)
    axs[2].grid(which = "major", axis = "both", linestyle = '-', \
                color = "gray", linewidth = 0.5)
    if isembedded:
        return fig, axs
    else:
        plt.show()
        return None

def Describe_Performance(asset_record_df, trade_record_df, riskfreerate = 0.):
    """
    Trade performance summary
    Arguments:
    (1) asset_record_df with columns
        - current iteration
        - previous avg. stock price
        - current asset = cash + stock value
        - current cash
        - current stock value
        - current action
        - current position (after action)
    (2) trade_record_df with columns
        - long or short (-)
        - quantities (in shares)
        - entry price (NT$/share)
        - exit price (NT$/share)
        - hold period (days)
        - profit and loss; PnL (NT$)
        - total fees (NT$)
        - net PnL / reward of this trade (NT$)
    (3) risk-free rate, default = 0.
    Return: 
        dict contains performance indicators
        current version: 22 items in the dict.
    (1) basic info:
        - test_days (days)
        - initial_cash (NT$)
    (2) earning and risk
        - reward (at the end of test_days; in NT$)
        - max_underwater_time (max period asset below initial value; in days)
        - max_underwater_ratio (max loss based on initial asset value; in %)
        - max_drawdown_ratio (max loss based on previous maximum asset value; in %)
    (3) return based on the initial asset value
        - return (at the end of test_days; in %)
        - annual_return (at the end of test_days; in %)
        - max_return (max. return based on initial asset value; in %)
        - min_return (min. return based on initial asset value; in %)
    (4) performance based on risk (volatility) taken
        - annual_volatility (return basis, so in %)
        - sharpe (Sharpe ratio with default risk-free rate = 0)
        - sortino (modified Sharpe ratio with default risk-free rate = 0)
    (5) trading info
        - max_hold_days (max hold period of all trades; in days)
        - no_trades (number of trades)
        - avg_hold_days (average hold period of each trade; in days)
        - cover_ratio (percentage of days hold position; in %)
    (6) profitability of single trade
        - max_profit (max profit of single trade; in NT$)
        - min_profit (min profit of single trade; in NT$)
        - avg_profit (average profit of each trade; in NT$)
    (7) overall trading performance
        - gain_pain_ratio (sum of profit / |sum of loss|; in %)
        - win_rate (percentage of trades that gains money; in %)
    """
    perf_dict = {}
    x = asset_record_df.iteration.values
    y1 = asset_record_df.cash_and_stock.values
    z1 = trade_record_df.hold_period.values
    z2 = trade_record_df.net_profit_and_loss.values
    # z3 = trade_record_df.profit_and_loss.values
    
    y1_s = asset_record_df.cash_and_stock
    y1_s0 = asset_record_df.cash_and_stock.shift(1)
    ret = (y1_s/y1_s0 - 1.0) * 100. # daily return
    
    # Analysis based on the asset record:
    # (1) basic info
    perf_dict["test_days"] = x[-1] - x[0]
    perf_dict["initial_cash"] = y1[0]
    
    # (2) earning and risk
    perf_dict["reward"] = y1[-1] - y1[0] # in NT$
    uwt_max = 0
    uwt = 0
    for i in range(len(y1)):
        if y1[i] < y1[0]: # 若資產減損，則累計蒙受虧損之天數
            uwt += 1
        else: # 若資產增值，更新最長天數紀錄後，重新起算下一次連續虧損天數
            uwt_max = max(uwt_max, uwt)
            uwt = 0
    uwt_max = max(uwt_max, uwt) # 最後再比較一次虧損天數
    perf_dict["max_underwater_time"] = uwt_max # in days
    perf_dict["max_underwater_ratio"] = min(min(y1 - y1[0]), 0) / y1[0] * 100.
    tmpdp = peak = 0. # initial value
    for i in range(1, len(y1)): # 記錄歷史資產最大值
        peak = max(y1[0: i])
        tmpdp = min(tmpdp, ((y1[i] - peak) / peak))
    # max draw-down ratio = (trough - peak)/peak*100%
    perf_dict["max_drawdown_ratio"] = tmpdp * 100.
    
    # (3) return based on initial asset value
    perf_dict["return"] = (y1[-1]/y1[0] - 1.)*100.
    perf_dict["annual_return"] = 100. * \
        ((y1[-1]/y1[0]) ** (myParams.DaysPerYear/perf_dict["test_days"]) - 1.)
    perf_dict["max_return"] = (max(y1)/y1[0] - 1.)*100.
    perf_dict["min_return"] = (min(y1)/y1[0] - 1.)*100.
    
    # (4) performance based on risk (volatility) taken
    perf_dict["annual_volatility"] = \
        np.sqrt(myParams.OpenDaysPerYear) * ret.std()
    expected_ret = ret.mean()
    volatility = ret.std()
    if volatility != 0.:
        perf_dict["sharpe"] = np.sqrt(myParams.OpenDaysPerYear) * \
            (expected_ret - riskfreerate) / volatility
    else:
        perf_dict["sharpe"] = 0.
    neg_rets = [r for r in ret if r < 0]
    neg_vol = np.std(neg_rets)
    if neg_vol != 0.:
        perf_dict["sortino"] = np.sqrt(myParams.OpenDaysPerYear) * \
            (expected_ret - riskfreerate) / neg_vol
    else:
        perf_dict["sortino"] = 0.
    
    # analysis based on the trade record:
    if len(z1) > 0:
        # (5) trading info
        # 注意：未平倉的交易不會被計入分析
        # 對 max_hold_days, avg_hold_days, cover_ratio 可能有影響
        perf_dict["max_hold_days"] = max(z1)
        perf_dict["no_trades"] = len(z1)
        perf_dict["avg_hold_days"] = np.mean(z1)
        perf_dict["cover_ratio"] = np.sum(z1)/perf_dict["test_days"] * 100.
        
        # (6) profitability of single trade
        perf_dict["max_profit"] = max(z2)
        perf_dict["min_profit"] = min(z2)
        perf_dict["avg_profit"] = np.mean(z2)
        
        # (7) overall trading performance
        gain = np.array([myParams.Tiny])
        pain = np.array([myParams.Tiny])
        if perf_dict["max_profit"] > 0.:
            gain = [z for z in z2 if z > 0]
        if perf_dict["min_profit"] < 0:
            pain = [-z for z in z2 if z <= 0]
        if np.sum(pain) <= 1e-4:
            perf_dict["gain_pain_ratio"] = 1e6
        else:
            perf_dict["gain_pain_ratio"] = \
                np.sum(gain) / np.sum(pain) * 100.
        if perf_dict["max_profit"] > 0.:
            perf_dict["win_rate"] = len(gain)/len(z2) * 100.
        else:
            perf_dict["win_rate"] = 0.
    else: # 無任何交易記錄
        perf_dict["max_hold_days"] = 0
        perf_dict["no_trades"] = 0
        perf_dict["avg_hold_days"] = 0.
        perf_dict["cover_ratio"] = 0.
        perf_dict["max_profit"] = 0.
        perf_dict["min_profit"] = 0.
        perf_dict["avg_profit"] = 0.
        perf_dict["gain_pain_ratio"] = 0.
        perf_dict["win_rate"] = 0.
    return perf_dict

# Done!
