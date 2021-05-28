# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from statsmodels.tsa import stattools
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
import scipy.stats as stats
from arch.unitroot import ADF # Augmented Dickey-Fuller 定態檢定
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

"""
myTSTests,py 所具有的功能：
(1) 時間序列特徵擷取與檢定
(2) 時間序列特徵圖像化
(1)用於客觀的檢定分析，(2)則為(1)的對應作圖，供使用者主觀判斷
"""


RefPath = '' # 參考路徑
if __name__ == "__main__": # executed
    RefPath = os.path.dirname(os.getcwd())
    LogName = None # 當此腳本被執行時為根日誌(不設名稱)
else: # imported
    modulepath = os.path.dirname(__file__)
    RefPath = os.path.dirname(modulepath)

# Global variables and constants
ImgExt = ".png"
PVal = 0.05 # significance level (may be 0.01, 0.05, or 0.10)
Dest = RefPath # 輸出圖片之資料夾位置

# ---   ---
# 分析計算工具
def UnitRootTest(ret, isdefaultmethod = True, isprintsummary = False):
    """
    進行 Augmented Dickey-Fuller (ADF) 單根檢定
    若接受虛無假設則表示有單根，序列為非定態 (non-stationary)，回傳值為False
    若拒絕虛無假設則表示無單根，序列為弱定態 (weakly stationary)，回傳值為True
    """
    # 除非進行除錯，否則建議參數使用預設值
    if isdefaultmethod: # default & recommended in the current version
        results = ADF(ret, lags = None, max_lags = None, \
                      trend = 'c', method = "AIC")
        if isprintsummary:
            print(results.summary().as_text())
        if results.pvalue < PVal: # ADF測試對應p-value小於5%
            if isprintsummary:
                print("- - - - - - - - - - - - -  - - - - - - - - - - - - -")
                print("ADF stats p-value < %.2f -> weakly stationary!" % PVal)
                print("- - - - - - - - - - - - -  - - - - - - - - - - - - -")
            return True
        else:
            if isprintsummary:
                print("- - - - - - - - - - - - -  - - - - - - - - - - - - -")
                print("ADF stats p-value > %.2f -> non-stationary!" % PVal)
                print("- - - - - - - - - - - - -  - - - - - - - - - - - - -")
            return False
    else: # future preference
        results = stattools.adfuller(ret, maxlag = None, \
                                     regression = 'c', autolag = "AIC", \
                                     store = False, regresults = False)
        # returns: adfstat, pvalue, usedlag, nobs, critvalues(, icbest)
        if isprintsummary:
            print("ADF test statistics: ", results[0])
            print("MacKinnon approximated p-value: ", results[1])
            print("# of lags used: ", results[2])
            print("# of obs. used for ADF test: ", results[3])
            print("Critical values for p-value = 0.01, 0.05, and 0.10: ")
            print(results[4]["1%"])
            print(results[4]["5%"])
            print(results[4]["10%"])
            print("The best information criterion (min. AIC): ", results[5])
        if results[1] < PVal: # ADF測試對應p-value小於5%
            if isprintsummary:
                print("- - - - - - - - - - - - -  - - - - - - - - - - - - -")
                print("ADF stats p-value < %.2f -> weakly stationary!" % PVal)
                print("- - - - - - - - - - - - -  - - - - - - - - - - - - -")
            return True
        else:
            if isprintsummary:
                print("- - - - - - - - - - - - -  - - - - - - - - - - - - -")
                print("ADF stats p-value > %.2f -> non-stationary!" % PVal)
                print("- - - - - - - - - - - - -  - - - - - - - - - - - - -")
            return False
    # return None

def WhiteNoiseTest(ret, nlags = 20, isprintsummary = False):
    """
    白雜訊測試：使用 Ljung-Box 檢定
    若接受虛無假設則表示為白雜訊(純隨機序列)，回傳值為False
    若拒絕虛無假設則表示序列並非隨機的，回傳值為True
    """
    acf = stattools.acf(ret, nlags = nlags, qstat = False, \
                        fft = True, alpha = None, missing = "drop")
    n = len(ret)
    results = stattools.q_stat(acf, n)
    if isprintsummary:
        print("Show Ljung-Box Q-statistics: ", results[0])
        print("Show corresponding P-values: ", results[1])
    if np.any(results[1] < PVal): # 只要有一Q統計量顯著大於0，即滿足條件
        if isprintsummary:
            print("- - - - - - - - - - - - -  - - - - - - - - - - - - -")
            print("Some LB stats p < %.2f -> correlated time series!" % PVal)
            print("- - - - - - - - - - - - -  - - - - - - - - - - - - -")
        return True
    else: # All p-values > PVal, accept H0 (white noise time series)
        if isprintsummary:
            print("- - - - - - - - - - - - -  - - - - - - - - - - - - -")
            print("All LB stats p > %.2f -> white noise time series!" % PVal)
            print("- - - - - - - - - - - - -  - - - - - - - - - - - - -")
        return False
    # return None

def ARCHEffectTest(ret,  nlags = 20, isreturndetails = False, \
                   isprintsummary = False):
    """
    波動叢聚 (volatility clustering) 測試，或檢測是否具有 ARCH 效應
    """
    if type(ret) is np.ndarray: # numpy ndarray
        ret_abs = np.absolute(ret)
        ret_sq = ret**2
    else: # pandas dataframe or series; if not, raise error
        ret_abs = ret.abs()
        ret_sq = ret**2
    test = [False]*3
    # 針對ret進行檢定
    test[0] = WhiteNoiseTest(ret, nlags, isprintsummary)
    # 針對|ret|進行檢定
    test[1] = WhiteNoiseTest(ret_abs, nlags, isprintsummary)
    # 針對(ret)^2進行檢定
    test[2] = WhiteNoiseTest(ret_sq, nlags, isprintsummary)
    if isprintsummary:
        print("====================================================")
        print("ARCH effect (volatility clustering) testing: ")
        print("Ljung-Box test results (true if correlated):")
        print("ret  ret_abs  ret_sq (expected: False / True / True)")
        print("%s  %s  %s" % (test[0], test[1], test[2]))
        print("====================================================")
    if test[0] is False and test[1] is True and test[2] is True:
        if isreturndetails:
            return [True, test[0], test[1], test[2]] # 總結果與3次LB測試結果
        else:
            return True # 滿足ARCH效應或波動叢聚效應
    else:
        if isreturndetails:
            return [False, test[0], test[1], test[2]] # 總結果與3次LB測試結果
        else:
            return False
    # return None

def KSTest_Norm(ret, isprintsummary = False):
    """
    Kolmogorov-Smirnov test 用於辨別：
    一經驗分布與一理論分布(在此限定為常態分布)是否具有顯著差異
    """
    results = stats.kstest(ret, "norm", N = len(ret), \
                           alternative = "two-sided", mode = "auto")
    if isprintsummary:
        print("- - - - - - - - - - - - -  - - - - - - - - - - - - -")
        print("K-S test results: (significance level: %.2f)" % PVal)
        print("K-S D = %.2e, p-value = %.2e." % (results[0], results[1]))
    if results[1] < PVal:
        if isprintsummary:
            print("The input return is NOT normal distribution!")
            print("- - - - - - - - - - - - -  - - - - - - - - - - - - -")
        return True # 拒絕虛無假設，即ret不為常態分布
    else:
        if isprintsummary:
            print("The input return is normal distribution!")
            print("- - - - - - - - - - - - -  - - - - - - - - - - - - -")
        return False # 接受虛無假設，即ret為常態分布
    # return None

# 作圖顯示
def GenQQPlot(ret, fname, issaveplots = False, isshowplots = False):
    """
    產生報酬率數據之 quantile-quantile plot
    (使用scipy.stats模組)
    ※縱軸為排序後之實際值，而非百分數，這點與statsmodels模組不同
    """
    # matplotlib default settings:
    mpl.style.use("classic")
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    res = stats.probplot(ret, dist = stats.norm, \
                         fit = True, plot = ax)
    ax.set_title("Q-Q plot, return vs. fitted gaussian dist.")
    if issaveplots:
        fpath = Dest + "\\qqplot-" + fname + ImgExt
        fig.savefig(fpath)
    if isshowplots:
        plt.show()
    return None

def GenQQPlot2(ret, fname, isfit = True, isregline = False, \
               issaveplots = False, isshowplots = False):
    """
    產生報酬率數據之 quantile-quantile plot
    (使用statsmodels模組，是較建議的選擇)
    若 isfit = True 則 loc 與 scale 兩參數不會用到
    若 isfit = False 則與標準常態分布作比較(loc = 0, scale = 1)
    """
    # matplotlib default settings:
    mpl.style.use("classic")
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    line = '45' # default: 45-degree line (y = x)
    if isregline:
        line = 'r' # least-square regression line to the sample data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    res = qqplot(ret, dist = stats.norm, fit = isfit, \
                 loc = 0., scale = 1., line = line, ax = ax)
    ax.set_title("Q-Q plot, return vs. gaussian distribution")
    if issaveplots:
        fpath = Dest + "\\qqplot2-" + fname + ImgExt
        fig.savefig(fpath)
    if isshowplots:
        plt.show()
    return None

def GenACF(ret, fname, maxlag = None, issaveplots = False, \
           isshowplots = False):
    """
    產生報酬率數據之 auto-correlation function
    預設待計算的時間序列很長，故使用快速傅立葉轉換(FFT)處理卷積分
    maxlag 可為技術指標中使用的最長天期，例如60天
    """
    # matplotlib default settings:
    mpl.style.use("classic")
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    # nlags 表示落後期數，在此建議用預設交易期間天數(trade day)
    if maxlag is None:
        nlags = None # use default len(corr)
    else:
        nlags = min(len(ret)/2, maxlag) # maxlag is used if it < len(ret)/2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 除了 ax, lags, fft, missing 參數外均使用預設值
    plot_acf(ret, ax = ax, lags = nlags, alpha = 0.05, use_vlines = True, \
             fft = True, missing = "drop")
    if issaveplots:
        fpath = Dest + "\\acf-" + fname + ImgExt
        fig.savefig(fpath)
    if isshowplots:
        plt.show()
    return None

def GenPACF(ret, fname, maxlag = None, issaveplots = False, \
            isshowplots = False):
    """
    產生報酬率數據之 partial auto-correlation function
    與ACF不同，PACF排除其他期對該期的影響
    """
    # matplotlib default settings:
    mpl.style.use("classic")
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    if maxlag is None:
        nlags = None # use default len(corr)
    else:
        nlags = min(len(ret)/2, maxlag) # maxlag is used if it < len(ret)/2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 除了 ax 與 lags 參數外均使用預設值
    plot_pacf(ret, ax = ax, lags = nlags, alpha = 0.05, use_vlines = True, \
              method = "yw")
    if issaveplots:
        fpath = Dest + "\\pacf-" + fname + ImgExt
        # print(fpath)
        fig.savefig(fpath)
    if isshowplots:
        plt.show()
    return None

def ShowVolCluster(ret, fname, maxlag = None, issaveplots = False, \
                   isshowplots = False):
    """ 將波動叢聚效應圖像化 """
    # matplotlib default settings:
    mpl.style.use("classic")
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    if maxlag is None:
        nlags = None # use default len(corr)
    else:
        nlags = min(len(ret)/2, maxlag) # maxlag is used if it < len(ret)/2
    if type(ret) is np.ndarray: # numpy ndarray
        ret_abs = np.absolute(ret)
        ret_sq = ret**2
    else: # pandas dataframe or series; if not, raise error
        ret_abs = ret.abs()
        ret_sq = ret**2
    # 繪製報酬率/報酬率絕對值/報酬率平方值之收益圖與ACF作圖：
    fig, axs = plt.subplots(3, 2, sharex = "col", constrained_layout = True)
    fig.suptitle('Volatility clustering effect')
    plot_acf(ret, ax = axs[0, 0], lags = nlags, alpha = 0.05, \
             use_vlines = True, fft = True, missing = "drop")
    plot_acf(ret_abs, ax = axs[1, 0], lags = nlags, alpha = 0.05, \
             use_vlines = True, fft = True, missing = "drop")
    plot_acf(ret_sq, ax = axs[2, 0], lags = nlags, alpha = 0.05, \
             use_vlines = True, fft = True, missing = "drop")
    axs[0, 0].set_title("ACF of return", )
    axs[1, 0].set_title("ACF of absolute return")
    axs[2, 0].set_title("ACF of squared return")
    axs[0, 1].plot(ret)
    axs[0, 1].set_title("Return")
    axs[1, 1].plot(ret_abs)
    axs[1, 1].set_title("Absolute return")
    axs[2, 1].plot(ret_sq)
    axs[2, 1].set_title("Squared return")
    # 對報酬率作圖設定日期格式，縮短顯示標籤長度避免標籤互相重疊影響美觀
    for nn, ax in enumerate(axs[:, 1]):
        locator = mdates.AutoDateLocator(minticks = 4, maxticks = 10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    # 圖像輸出
    if issaveplots:
        fpath = Dest + "\\vol_cluster-" + fname + ImgExt
        fig.savefig(fpath)
    if isshowplots:
        plt.show()
    return None

if __name__ == "__main__":
    X = "2330.TW_demo_v1"
    fpath = RefPath + '\\' + X + ".xlsx"
    df_X = pd.read_excel(fpath, header = 0, index_col = 0, \
                         skiprows = None)
    ret = df_X["R-C"]
    ret.index = pd.to_datetime(ret.index)
    ffname = X
    
    # 對報酬率數據進行各種測試分析
    isprintmsg = True
    n_lags = 30
    isshowplots = True
    issaveplots = True
    
    # 常態分布檢定
    t1 = KSTest_Norm(ret[1:], isprintmsg)
    print("Norm-dist test results: ", t1)
    GenQQPlot(ret[1:], ffname, issaveplots, isshowplots)
    GenQQPlot2(ret[1:], ffname, True, False, issaveplots, isshowplots)
    
    # 定態檢定
    t2 = UnitRootTest(ret[1:], True, isprintmsg)
    # t2 = UnitRootTest(ret[1:], False, isprintmsg) # 與以上結果相同
    print("Stationary test results: ", t2)
    
    # 白雜訊檢定
    t3 = WhiteNoiseTest(ret[1:], n_lags, isprintmsg)
    print("White noise test results: ", t3)
    GenACF(ret[1:], ffname, n_lags, issaveplots, isshowplots)
    GenPACF(ret[1:], ffname, n_lags, issaveplots, isshowplots)
    
    # 波動叢聚/ARCH效應檢定
    t4 = ARCHEffectTest(ret[1:],  n_lags, True, isprintmsg)
    print("Volatility clustering test results: ", t4)
    ShowVolCluster(ret[1:], ffname, n_lags, issaveplots, isshowplots)
# Done!
