# -*- coding: utf-8 -*-

import os
from math import sqrt, log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

from myUtilities import myParams
from myUtilities import myLog

"""
myTechIndices.py的功能：
(1) 技術指標計算
(2) 運用mplfinance繪製基本K線圖(價格-成交量)
(3) 運用mplfinance繪製進階K線圖(最多一個主圖與兩個附圖技術指標)
"""

# Global variables and constants:
plt.set_loglevel("warning") # avoid log polluted by mpl

DefTIList = myParams.DefinedTechInd # 排序後: KD, MACD, SMA, STD, VIX, VWMA
BaseColumns = ["V", "O", "H", "L", "C"]
BaseColForTA = ["volume", "open", "high", "low", "close"]
KW_R = "R-"

IsFillNA = False # 是否填滿技術指標計算時產生的NaN？(建議設False)

AppliedTIs = {}
IsReturnCalc = False # 是否計算技術指標的日報酬率

### 隨機指標
# KD & Fast-KD 參數
RSV_Span = 9
K_Span = 3
D_Span = 3
KDBaseP = 'C' # default 'C', may be 'O'
K0 = 50 # 用來計算第一筆Kt之Kt-1預設值，第一筆K值出現於RSV_Span當天
D0 = 50 # 用來計算第一筆Dt之Dt-1預設值，第一筆D值出現於RSV_Span當天

### 趨勢指標
# MACD 參數
DIF_Fast_Span = 12
DIF_Slow_Span = 26
DEF_Span = 9
MACDBaseP = 'C' # Default 'C', may be 'D' = demand index
IsDEFUseEMA = True # If False, apply SMA
# MACDBaseP = 'D' # Default 'C', may be 'D' = demand index
# IsDEFUseEMA = False # If False, apply SMA

# SMA 參數
SMABaseP = 'C' # Default 'C', may be 'D' = demand index

# VWMA 參數
VWMABaseP = 'D' # Default 'D' = demand index, may be 'C'
VWMABaseW = 'M' # Default 'M' = monetary, may be 'S' = shares
# VWMABaseP = 'C' # Default 'D' = demand index, may be 'C'
# VWMABaseW = 'S' # Default 'M' = monetary, may be 'S' = shares

### 動能指標(較推薦優先用CVOL，另外兩者波型和靈敏度不同，亦可併用)
# STD 參數
STDBaseP = 'C' # Default = 'C'; may be 'D' = demand index
IsLogPForSTD = False # Use logP or P (建議為False)

# VIX 參數
VIXBaseP = 'D' # Default 'D' = demand index, may be 'C'

# CVOL 參數：
CVOLRType = 0 # 0: High - Low (預設); 1: true range

# 視窗作圖
Dpath = ''
if __name__ == "__main__":
    Dpath = os.path.dirname(os.getcwd())
else:
    modulepath = os.path.dirname(__file__)
    Dpath = os.path.dirname(modulepath)

# 不同時間週期繪圖的配色(用於SMA, VIX等技術指標)
MA_Colors = ['y', 'c', 'm', 'b', 'g', 'r']

# *** 參數資訊輸出 ***
def ShowTIParams(logger = None):
    """ TI模組技術指標參數輸出 (2020.11.12 更新) """
    kdparams = [RSV_Span, K_Span, D_Span, KDBaseP, K0, D0]
    macdparams = [DIF_Fast_Span, DIF_Slow_Span, DEF_Span, \
                  MACDBaseP, IsDEFUseEMA]
    smaparams = [SMABaseP]
    vwmaparams = [VWMABaseP, VWMABaseW]
    stdparams = [STDBaseP, IsLogPForSTD]
    vixparams = [VIXBaseP]
    cvolparams = [CVOLRType]
    if logger is None:
        print("是否處理缺失值: ", IsFillNA)
        print("價格：OHLC: 開盤/最高/最低/收盤價, D: 需求指數(DI)")
        print("量能：V: 成交量, M: 成交金額")
        print("KD參數：RSV天期, K天期, D天期, 價格, 初始K值, 初始D值\n", \
              kdparams)
        print("MACD參數：DIF短天期, DIF長天期, DEF天期, 價格, 是否用EMA\n", \
              macdparams)
        print("SMA參數：價格\n", smaparams)
        print("VWMA參數：價格, 量能\n", vwmaparams)
        print("STD參數：價格, 是否對數價格\n", stdparams)
        print("VIX(歷史波動率)參數：價格\n", vixparams)
        print("CVOL(查金氏波動指標)參數：天期 = (7, 14, 28), ")
        if cvolparams[0] == 0:
            print("價格採用 range = Ht - Lt")
        else:
            print("價格採用 true range = max(Ht, Ct-1) - min(Lt, Ct-1)")
    else:
        logger.info("是否處理缺失值: %s" % IsFillNA)
        logger.info("價格：OHLC: 開盤/最高/最低/收盤價, D: 需求指數(DI)")
        logger.info("量能：V: 成交量, M: 成交金額")
        logger.info("KD參數：RSV天期, K天期, D天期, 價格, 初始K值, 初始D值")
        logger.info("{}".format(', '.join(map(str, kdparams))))
        logger.info("MACD參數：DIF短天期, DIF長天期, DEF天期, 價格, 是否用EMA")
        logger.info("{}".format(', '.join(map(str, macdparams))))
        logger.info("SMA參數：價格 %s" % smaparams[0])
        logger.info("VWMA參數：價格 %s, 量能 %s" % \
                    (vwmaparams[0], vwmaparams[1]))
        logger.info("STD參數：價格 %s, 是否對數價格 %s" % \
                    (stdparams[0], stdparams[1]))
        logger.info("VIX(歷史波動率)參數：價格 %s" % vixparams[0])
        logger.info("CVOL(查金氏波動指標)參數：天期 = (7, 14, 28)")
        if cvolparams[0] == 0:
            logger.info("價格採用 range = Ht - Lt")
        else:
            logger.info("價格採用 true range = max(Ht, Ct-1) - min(Lt, Ct-1)")
    return None

# *** 初始化函式 ***
def IsValidBaseDF(cols):
    """ 簡單驗證是否為基本DataFrame """
    valid_df = False
    extracted = [item for item in cols if item in BaseColumns]
    if set(extracted) == set(BaseColumns):
        valid_df = True
    return valid_df

def InitDF(df0, isbasedf = True):
    """
    對歷史資料的欄位做初步檢查
    若使用TA-LIB的abstract API，則欄位名稱需更改
    """
    if IsValidBaseDF(df0.columns) and isbasedf == True:
        df = df0.copy()
        return df
    return df0

def TISetUp(tilist, isdr):
    """ 技術指標計算基本設定 """
    global AppliedTIs, IsReturnCalc
    for kw in DefTIList: # 新建或重設包含所有已定義指標的字典
        AppliedTIs.update({kw: False})    
    for kw in DefTIList:
        if kw in tilist: # 被指定使用的技術指標
            AppliedTIs.update({kw: True})
    # 對於上述選取的技術指標是否計算日報酬率
    IsReturnCalc = bool(isdr) # no validation
    return None

# *** KD相關函式 ***
def Simple_EMA(p, old, new):
    """ EMA = (1-p)/p*old + 1/p*new = old + 1/p*(new - old) """
    return old + 1/p*(new - old)

def GetK(rsv):
    """ Kt = 2/3*Kt-1 + 1/3*RSVt """
    global K0
    if not np.isnan(rsv): # start to calc. K if RSV is available
        K0 = Simple_EMA(K_Span, K0, rsv)
        return K0
    else:
        return np.nan

def GetD(k):
    """ Dt = 2/3*Dt-1 + 1/3*Kt """
    global D0
    if not np.isnan(k): # start to calc. D if K is available
        D0 = Simple_EMA(D_Span, D0, k)
        return D0
    else:
        return np.nan

def KD(df0, cols):
    """
    KD指標計算：快速K線、K線(快速D線)、D線
    TA-LIB的slow-K和slow-D計算結果與自行撰寫的相差甚大，
    然而自行計算結果與XQ系統和超級大三元接近
    """
    if not AppliedTIs["KD"]: # If dict["KD"] is True
        return pd.DataFrame(data = [], columns = [])
    df = InitDF(df0, True)
    # 價位允許使用close或open price
    global K0, D0, KDBaseP
    if KDBaseP != 'C' and KDBaseP != 'O':
        KDBaseP = 'C'
    MaxP = df['H'].rolling(RSV_Span).max()
    minP = df['L'].rolling(RSV_Span).min()
    df[cols[0]] = (df[KDBaseP] - minP)/(MaxP - minP)*100.0 # Fast-K
    df[cols[1]] = df[cols[0]].apply(GetK) # Slow-K
    df[cols[2]] = df[cols[1]].apply(GetD) # Slow-D
    if IsFillNA:
        df.fillna(50.0, inplace = True) # 用中性值填滿缺失值
    K0 = D0 = 50 # Reset global variables
    return df[cols]

# *** 均線指標(包含MACD)計算相關函式 ***
def GetDI(df):
    """ Calculate demand index (DI) = (H + L + 2C)/4 """
    s_di = (df['H'] + df['L'] + 2.0*df['C'])/4.0
    s_di.name = 'D'
    return s_di

def WSMA(tp, s_p, s_w = None):
    """ 
    用於簡單移動平均(SMA)，或加權簡單移動平均(WMA)計算
    tp: 期間
    """
    # No validation for tp, s_p, s_w here
    if s_w is None:
        s_wsma = s_p.rolling(tp).mean()
    else:
        s_wsma = (s_p*s_w).rolling(tp).sum()/s_w.rolling(tp).sum()
    return s_wsma

def EMA(tp, s_p, IsAddOne = False):
    """
    用於指數移動平均(EMA)計算
    EMAt = EMAt-1 + alpha*(Pt - EMAt-1)
    alpha = (1 + a)/(tp + a)
    KD指標計算: alpha = 1/3 (tp = 3, a = 0)
    MACD-DIF: alpha = 2/13 or 2/27 (tp = 12 or 26, a = 1)
    當天期 = tp 時，由於沒有EMA前值，故當日以SMA取代
    """
    a = 1 if IsAddOne else 0
    alpha = (1 + a)/(float(tp) + a)
    beta = 1.0 - alpha
    s_ema = WSMA(tp, s_p, None) # 先全部用SMA算，之後再更新
    counts = 0
    for i in range(tp, s_p.shape[0]):
        # 資料驗證(若s_p為EMA計算結果，天期的基準點必須調整)
        # 例如計算DEF時，第一筆可計算的資料為第34筆而非第9筆
        if not np.isnan(s_ema.iat[i - 1]) or counts > 0:
            s_ema.iat[i] = alpha*s_p.iat[i] + beta*s_ema.iat[i - 1]
            counts += 1
            # 開始計算之後遇到NaN也無所謂，繼續算
    return s_ema

def MACD(df0, cols):
    """ MACD計算：DIF、DEF(MACD)、MACD柱狀體(Bar或OSC) """
    if not AppliedTIs["MACD"]:
        return pd.DataFrame(data = [], columns = [])
    df = InitDF(df0, True)
    # 價位允許使用close或demand index
    # DEF的計算可以用SMA或EMA
    global MACDBaseP
    if MACDBaseP != 'C' and MACDBaseP != 'D':
        MACDBaseP = 'C' # 預設值
    s_p = df['C']
    if MACDBaseP == 'D': # 需求指數(demand index)
        s_p = GetDI(df)
    s_ema_f = EMA(DIF_Fast_Span, s_p, True)
    s_ema_s = EMA(DIF_Slow_Span, s_p, True)
    df[cols[0]] = s_ema_f - s_ema_s # DIF
    if IsDEFUseEMA:
        df[cols[1]] = EMA(DEF_Span, df[cols[0]], True) # DEF
    else:
        df[cols[1]] = WSMA(DEF_Span, df[cols[0]], None) # DEF
    df[cols[2]] = df[cols[0]] - df[cols[1]] # MACD Bar
    if IsFillNA:
        df.fillna(0.0, inplace = True)
    return df[cols]

def SMA(df0, cols, dpset):
    """ 簡單移動平均計算，dpset表示計算天期清單 """
    if not AppliedTIs["SMA"]:
        return pd.DataFrame(data = [], columns = [])
    df = InitDF(df0, True)
    i = 0
    # 價位允許使用：close 或 demand index
    global SMABaseP
    if SMABaseP != 'C' and SMABaseP != 'D':
        SMABaseP = 'C' # 預設值
    s_p = df['C']
    if SMABaseP == 'D': # 需求指數(demand index)
        s_p = GetDI(df)
    for dp in dpset:
        df[cols[i]] = WSMA(dp, s_p, None)
        i += 1
    if IsFillNA:
        df.fillna(0.0, inplace = True)
    return df[cols]

def VWMA(df0, cols, dpset):
    """
    成交量加權移動平均計算(自訂指標)
    允許使用成交股數或成交金額做權重，
    其中成交金額估算法為 成交股數 x 代表價格(收盤價或需求指數)
    建議需求指數搭配成交金額，或收盤價搭配成交股數使用
    """
    if not AppliedTIs["VWMA"]:
        return pd.DataFrame(data = [], columns = [])
    df = InitDF(df0, True)
    df.columns = BaseColumns
    i = 0
    # 價位允許使用：close 或 demand index
    global VWMABaseP, VWMABaseW
    if VWMABaseP != 'C' and VWMABaseP != 'D':
        VWMABaseP = 'D' # 預設值與SMA不同
    s_p = GetDI(df) if VWMABaseP == 'D' else df['C']
    # 權重允許使用：股數share或金額monetary
    if VWMABaseW != 'M' and VWMABaseW != 'S':
        VWMABaseW = 'M' # 預設值
    s_w = df['V'] # if VWMABaseW == 'S'
    if VWMABaseW == 'M': # 成交金額
        s_w = df['V']*s_p
        s_w.name = 'M'
    # 針對不同天期個別計算，預設dpset包含三種不同長短天期
    for dp in dpset:
        df[cols[i]] = WSMA(dp, s_p, s_w)
        i += 1
    if IsFillNA:
        df.fillna(0.0, inplace = True)
    return df[cols]

# *** 波動幅度(動能)估算相關函式 ***
def STD(df0, cols, dpset):
    """
    成交價標準差，建議使用需求指數之自然對數
    或是為求簡化直接使用收盤價
    """
    if not AppliedTIs["STD"]:
        return pd.DataFrame(data = [], columns = [])
    df = InitDF(df0, True)
    df.columns = BaseColumns
    i = 0
    # 價位允許使用：close 或 demand index
    global STDBaseP, IsLogPForSTD
    if STDBaseP != 'C' and STDBaseP != 'D':
        STDBaseP = 'C'
    s_p = GetDI(df) if STDBaseP == 'D' else df['C']
    if IsLogPForSTD: # 使用自然對數價位
        s_p = s_p.apply(log)
    # 針對不同天期個別計算，預設dpset包含三種不同長短天期
    for dp in dpset:
        df[cols[i]] = s_p.rolling(dp).std(skipna = True)
        i += 1
    if IsFillNA:
        df.fillna(0.0, inplace = True)
    return df[cols]

def VIX(df0, cols, dpset):
    """
    歷史波動率，在此以年化報酬率標準差之百分比表示
    價格會取自然對數，因 ln( 1 + rt ) ~ rt 之故
    rt 為報酬率 = (Pt - Pt-1)/Pt-1 x 100%
    """
    if not AppliedTIs["VIX"]:
        return pd.DataFrame(data = [], columns = [])
    df = InitDF(df0, True)
    df.columns = BaseColumns
    i = 0
    # 價位允許使用：close 或 demand index
    global VIXBaseP
    if VIXBaseP != 'C' and VIXBaseP != 'D':
        VIXBaseP = 'D' # 預設值與SMA不同
    s_p = GetDI(df) if VIXBaseP == 'D' else df['C']
    s_x = (s_p/s_p.shift(1)).apply(log) # 自然對數日報酬率
    # 針對不同天期個別計算，預設dpset包含三種不同長短天期
    for dp in dpset:
        df[cols[i]] = s_x.rolling(dp).std(skipna = True)
        df[cols[i]] *= sqrt(myParams.OpenDaysPerYear)*100.0
        i += 1
    if IsFillNA:
        df.fillna(0.0, inplace = True)
    return df[cols]

def CVOL(df0, cols):
    """
    查金氏波動指標：Chaikin's Volatility (CV)
    https://www.yuantafutures.com.tw/ytf/YT_online_help/ftv20/6-1.htm
    https://xstrader.net/%E5%85%8B%E6%9E%97%E6%B3%A2%E5%8B%95%E6%8C%87%E6%A8%99-chaikin-volatility/
    利用區間振幅估算，反映的是期間內的動能
    建議週期為14天，在此固定為7, 14, 28天 (因會用到兩次，故偏好短天期)
    """
    if not AppliedTIs["CVOL"]:
        return pd.DataFrame(data = [], columns = [])
    df = InitDF(df0, True)
    df.columns = BaseColumns
    i = 0
    dps = sorted((7, 14, 28))
    if CVOLRType == 0: # Use simple method (default)
        TR = df['H'] - df['L'] # XQ與超級大三元使用的設定
    else: # Use true range
        C0 = df['C'].shift(1)
        # 計算 TrueRange = max(Ct-1, Ht) - min(Ct-1, Lt)
        TR = pd.concat([C0, df['H']], axis = 1).max(axis = 1) - \
            pd.concat([df['L'], C0], axis = 1).min(axis = 1)
    for dp in dps:
        # 各天期先求一次移動平均，再求n天期間的百分離差
        TRMA = EMA(dp, TR, True) # TR之指數移動平均
        df[cols[i]] = (TRMA/TRMA.shift(dp) - 1.0)*100.0
        i += 1
    if IsFillNA:
        df.fillna(0.0, inplace = True)
    return df[cols]

# *** 報酬率計算 ***
def IsColsMatched(cols, cols_r, IsSameLen = False):
    if IsSameLen and len(cols) != len(cols_r):
        return False
    cols_1 = [KW_R + item for item in cols]
    col_match = [item for item in cols_r if item in cols_1]
    return (True if col_match == cols_r else False)

def DROIP(s_num, s_dom, islogreturn = False):
    """ Daily Return Of Investment in Percentage """
    if islogreturn: # 金融實務常用；log(Pt/Pt-1) ~ Pt/Pt-1 - 1
        s_r = (s_num/s_dom).apply(log)*100.0
    else:
        s_r = ((s_num/s_dom) - 1.0)*100.0
    if IsFillNA:
        s_r.fillna(0.0, inplace = True)
    return s_r

def VolRatio(S_V):
    """ 昨量比 """
    S_VR = (S_V/S_V.shift(1))*100.0
    if IsFillNA:
        S_VR.fillna(100.0, inplace = True) # 預期相連天期成交量相同
    return S_VR

def PriceDailyReturn(df0, cols_r, iscalcvolratio = True):
    """ 價位日報酬率，以收盤價為計算基準 """
    if not IsColsMatched(df0.columns, cols_r, False):
        return df0
    df = df0.copy()
    if iscalcvolratio:
        df["R-V"] = VolRatio(df['V']) # 昨量比
    S_C0 = df['C'].shift(1) # 昨收價
    df["R-O"] = DROIP(df['O'], S_C0, False) # 開盤漲跌幅
    df["R-H"] = DROIP(df['H'], S_C0, False)
    df["R-L"] = DROIP(df['L'], S_C0, False)
    df["R-C"] = DROIP(df['C'], S_C0, False) # 即當日漲跌幅
    if IsFillNA:
        df.fillna(0.0, inplace = True)
    return df[cols_r] # cols_r = ['R-V', 'R-O', 'R-H', 'R-L', 'R-C']

def TI_DailyReturn(df_ti0, cols_r):
    """ 技術指標日報酬率 """
    cols = df_ti0.columns
    if not IsColsMatched(cols, cols_r, True):
        return df_ti0
    df_ti = df_ti0.copy()
    for i in range(0, len(cols_r)):
        df_ti[cols_r[i]] = DROIP(df_ti[cols[i]], df_ti[cols[i]].shift(1))
    if IsFillNA:
        df_ti.fillna(0.0, inplace = True)
    return df_ti[cols_r]

# *** 圖形顯示 ***
class TIGraphProp():
    """ Properties for technical indicator graphing """
    def __init__(self, calc_kw, ismain = False, dpset = None):
        self.calc_kw = calc_kw.upper()
        self.ismain = bool(ismain) # no validation
        self.dpset = dpset
        self.calc_val = pd.DataFrame(data = [], columns = [])
        self.ma_signal = pd.DataFrame(data = [], columns = [])
    def getvalues(self, df):
        if self.calc_kw in DefTIList and IsValidBaseDF(df.columns):
            if self.calc_kw == "KD":
                Cols_KD = ["KD-FK", "KD-K", "KD-D"]
                self.calc_val = KD(df, Cols_KD)
            elif self.calc_kw == "MACD":
                Cols_MACD = ["MACD-DIF", "MACD-DEM", "MACD-Bar"]
                self.calc_val = MACD(df, Cols_MACD)
            elif self.calc_kw == "SMA":
                Cols_SMA = ["SMA-" + str(s) for s in self.dpset]
                self.calc_val = SMA(df, Cols_SMA, self.dpset)
                self.set_ma_signal(df, self.dpset) # 扣抵位置
            elif self.calc_kw == "VWMA":
                Cols_VWMA = ["VWMA-" + str(s) for s in self.dpset]
                self.calc_val = VWMA(df, Cols_VWMA, self.dpset)
                self.set_ma_signal(df, self.dpset) # 扣抵位置
            elif self.calc_kw == "STD":
                Cols_STD = ["STD-" + str(s) for s in self.dpset]
                self.calc_val = STD(df, Cols_STD, self.dpset)
            elif self.calc_kw == "VIX":
                Cols_VIX = ["VIX-" + str(s) for s in self.dpset]
                self.calc_val = VIX(df, Cols_VIX, self.dpset)
            elif self.calc_kw == "CVOL":
                Cols_CVOL = ["CVOL-7", "CVOL-14", "CVOL-28"]
                self.calc_val = CVOL(df, Cols_CVOL)
    def set_ma_signal(self, df, dpset):
        """ 找出MA扣抵位置 """
        cols = []
        p = df["L"].min()*0.8
        df_tmp = df.copy()
        for i in sorted(dpset):
            col = self.calc_kw + "_Signal_" + str(i)
            cols.append(col)
            a = df.shape[0] - i
            b = df.shape[1] + len(cols) - 1
            df_tmp[col] = np.nan
            df_tmp.iat[a, b] = p # others are np.nan
        self.ma_signal = df_tmp[cols].copy()

class TIGraph_Mpf():
    """
    運用Mpf產生K線圖與技術指標作圖
    已定義的作圖參數：
    fig_size_x / fig_size_y = float
    """
    def __init__(self, df, dpset = None, **kwargs):
        if dpset is None: # default = (10, 20, 60)
            self.dpset = sorted(set([myParams.DaysList[1], 
                                     myParams.DaysList[2], 
                                     myParams.DaysList[3]]))
        else:
            self.dpset = dpset
        # 針對df並未作任何檢查和驗證，如格式錯誤則直接發生錯誤
        self.df = df
        self.val_main = pd.DataFrame(data = [])
        self.sig_main = pd.DataFrame(data = [])
        self.val_sub1 = pd.DataFrame(data = [])
        self.val_sub2 = pd.DataFrame(data = [])
        # 技術指標類別初始化
        self.kd = TIGraphProp("KD", )
        self.macd = TIGraphProp("MACD", )
        self.sma = TIGraphProp("SMA", True, self.dpset)
        self.vwma = TIGraphProp("VWMA", True, self.dpset)
        self.std = TIGraphProp("STD", False, self.dpset)
        self.vix = TIGraphProp("VIX", False, self.dpset)
        self.cvol = TIGraphProp("CVOL", )
        # 作圖參數
        self.fx = kwargs["fig_size_x"] if "fig_size_x" in kwargs else 16
        self.fy = kwargs["fig_size_y"] if "fig_size_y" in kwargs else 12
    def _setrun(self, **kwargs):
        global AppliedTIs, IsReturnCalc
        tilist_bk = AppliedTIs.copy() # 參數設定備份
        isdr_bk = IsReturnCalc # 參數設定備份
        TISetUp(myParams.DefinedTechInd, False) # 覆寫全域參數
        ti_list = [self.kd, self.macd, self.sma, self.vwma, \
                   self.std, self.vix, self.cvol]
        for ti in ti_list:
            if "ti_main" in kwargs:
                if kwargs["ti_main"].upper() == ti.calc_kw \
                    and ti.ismain is True:
                    ti.getvalues(self.df)
                    self.val_main = ti.calc_val.copy()
                    if ti.calc_kw == "SMA" or ti.calc_kw == "VWMA":
                        self.sig_main = ti.ma_signal
            if "ti_sub1" in kwargs:
                if kwargs["ti_sub1"].upper() == ti.calc_kw \
                    and ti.ismain is False:
                    ti.getvalues(self.df)
                    self.val_sub1 = ti.calc_val.copy()
            if "ti_sub2" in kwargs:
                if kwargs["ti_sub2"].upper() == ti.calc_kw \
                    and ti.ismain is False:
                    ti.getvalues(self.df)
                    self.val_sub2 = ti.calc_val.copy()
        AppliedTIs = tilist_bk.copy() # 還原參數至先前設定
        IsReturnCalc = isdr_bk # 還原參數至先前設定
    def show_basic(self, ftitle = "Stock"):
        """
        顯示基本K線圖，僅K線與成交量
        使用mplfinance實現之
        """
        dest = Dpath + '\\' + ftitle + ".png"
        mc = mpf.make_marketcolors(up = 'r', down = 'g', inherit = True)
        ms = mpf.make_mpf_style(base_mpf_style = "yahoo", \
                                marketcolors = mc)
        # 不支援中文字型，圖表文字需為英文
        mfargs = dict(type = "candle", volume = True, \
                      columns = ("O", "H", "L", "C", "V"), \
                      show_nontrading = False, returnfig = True, \
                      figsize = (self.fx, self.fy), \
                      title = ftitle, style = ms, ylabel = "Price", \
                      ylabel_lower = "Volume in shares")
        fig, axes = mpf.plot(self.df, **mfargs)
        mpf.show()
        fig.savefig(dest)
        return None
    def show_adv(self, ftitle = "Stock", **kwargs):
        """
        顯示進階K線圖，K線搭配一主圖指標與兩個副圖指標
        使用mplfinance實現之(panel method)
        已定義的作圖參數：
        ti_main = "SMA" / "VWMA"
        ti_sub1 = "KD" / "MACD" / "STD" / "VIX" / "CVOL"
        ti_sub2 = "KD" / "MACD" / "STD" / "VIX" / "CVOL"
        """
        self._setrun(**kwargs) # 進行參數驗證與指標計算
        dest = Dpath + '\\' + ftitle + ".png"
        mc = mpf.make_marketcolors(up = 'r', down = 'g', inherit = True)
        ms = mpf.make_mpf_style(base_mpf_style = "yahoo", \
                                marketcolors = mc, y_on_right = True)
        # 不支援中文字型，圖表文字需為英文
        mfargs = dict(type = "candle", volume = True, \
                      columns = ("O", "H", "L", "C", "V"), \
                      show_nontrading = False, returnfig = True, \
                      figsize = (self.fx, self.fy), \
                      title = ftitle, style = ms, ylabel = "Price", \
                      ylabel_lower = "Volume in shares", \
                      panel_ratios = (1, 1))
        aps = []
        if "ti_main" in kwargs:
            tidf = self.val_main.copy()
            isma = False
            if kwargs["ti_main"] == "SMA" or kwargs["ti_main"] == "VWMA":
                isma = True # 主圖技術指標為均線類型
            for i in range(0, tidf.shape[1]):
                aps.append(mpf.make_addplot(tidf.iloc[:, i], \
                                            color = MA_Colors[i]))
                if isma: # 補均線扣抵記號，配色同均線
                    aps.append(mpf.make_addplot(self.sig_main.iloc[:, i], \
                                                type = "scatter", \
                                                markersize = 200, \
                                                marker = '^', \
                                                color = MA_Colors[i]))
        pid = 1 # Panel ID = 0 for candlestick, 1 for volume
        if "ti_sub1" in kwargs: # pid = 2
            pid += 1
            ap2 = self._setsubti(self.val_sub1, kwargs["ti_sub1"], pid)
            for item in ap2:
                aps.append(item)
        if "ti_sub2" in kwargs: # pid = 3 (or 2 if ti_sub1 is not assigned)
            pid += 1
            ap3 = self._setsubti(self.val_sub2, kwargs["ti_sub2"], pid)
            for item in ap3:
                aps.append(item)
        if len(aps) > 0:
            if pid == 2:
                mfargs["panel_ratios"] = (2, 1, 1)
            elif pid == 3:
                mfargs["panel_ratios"] = (3, 1, 1, 1)
            fig, axes = mpf.plot(self.df, addplot = aps, **mfargs)
        else:
            fig, axes = mpf.plot(self.df, **mfargs)
        mpf.show()
        fig.savefig(dest)
        return None
    def _setsubti(self, dfti, ti_kw, pid):
        """ 副指標繪圖設定 """
        col = list(dfti.columns)
        ap = []
        if ti_kw == "KD":
            ap = [mpf.make_addplot(dfti[col[0]], color = 'y', panel = pid), 
                  mpf.make_addplot(dfti[col[1]], color = 'c', panel = pid), 
                  mpf.make_addplot(dfti[col[2]], color = 'm', panel = pid, \
                                   ylabel = "KD")]
        elif ti_kw == "MACD":
            col.append(str(col[2] + "-Up"))
            col.append(str(col[2] + "-Down"))
            dfti[col[3]] = np.nan
            dfti[col[4]] = np.nan
            for i in range(0, dfti.shape[0]): # 將柱狀圖用雙色表示
                if dfti.iat[i, 2] >= 0.0:
                    dfti.iat[i, 3] = dfti.iat[i, 2]
                else:
                    dfti.iat[i, 4] = dfti.iat[i, 2]
            ap = [mpf.make_addplot(dfti[col[0]], color = 'y', panel = pid), 
                  mpf.make_addplot(dfti[col[1]], color = 'c', panel = pid, \
                                   ylabel = "MACD"), 
                  mpf.make_addplot(dfti[col[3]], type = "bar", color = 'r', \
                                   width = 0.75, panel = pid), 
                  mpf.make_addplot(dfti[col[4]], type = "bar", color = 'g', \
                                   width = 0.75, panel = pid, \
                                   ylabel = "MACD")]
        elif ti_kw == "STD" or ti_kw == "VIX" or ti_kw == "CVOL":
            for i in range(0, dfti.shape[1]):
                ap.append(mpf.make_addplot(dfti.iloc[:, i], \
                                           color = MA_Colors[i], \
                                           panel = pid, ylabel = ti_kw))
        return ap
    def emptydfs(self):
        """ 清空DataFrame資料 """
        ti_list = [self.kd, self.macd, self.sma, self.vwma, \
                   self.std, self.vix, self.cvol]
        for ti in ti_list:
            ti.calc_val = pd.DataFrame(data = [], columns = [])
            ti.ma_signal = pd.DataFrame(data = [], columns = [])
        self.sig_main = pd.DataFrame(data = [], columns = [])
        self.val_main = pd.DataFrame(data = [], columns = [])
        self.val_sub1 = pd.DataFrame(data = [], columns = [])
        self.val_sub2 = pd.DataFrame(data = [], columns = [])
        self.df = pd.DataFrame(data = [], columns = [])

# Function testing 
if __name__ == "__main__":
    ### 使用爬蟲下載的檔案
    """
    fpath = Dpath + "\\2330_demo_yfin.csv"
    df0 = pd.read_csv(fpath, delimiter = ',', header = 0, \
                      index_col = None, skiprows = None)
    df = pd.DataFrame(data = [], columns = ['V', 'O', 'H', 'L', 'C'], \
                      index = df0["Date"].values)
    df.index = pd.to_datetime(df.index)
    df["V"] = df0["Volume"].apply(int).values
    df["O"] = df0["Open"].apply(float).values
    df["H"] = df0["High"].apply(float).values
    df["L"] = df0["Low"].apply(float).values
    df["C"] = df0["Close"].apply(float).values
    
    ti_graph = TIGraph_Mpf(df.iloc[:200, :], fig_size_x = 24, fig_size_y = 18)
    ti_graph.show_basic("TW2330_Basic")
    
    TISetUp(myParams.DefinedTechInd, False) # 覆寫全域參數
    ti_graph.show_adv("TW2330_Adv_SMA_KD_CVOL", ti_main = "SMA", \
                      ti_sub1 = "KD", ti_sub2 = "CVOL")
    # ti_graph.show_adv("TW2330_Adv_X_X_CVOL", ti_sub2 = "CVOL")
    """
    
    ### 換一個方式建立DataFrame: 使用後處理取得的檔案
    fpath = Dpath + "\\2330.TW_demo_v1.xlsx" 
    df0 = pd.read_excel(fpath, header = 0, index_col = None, skiprows = None)
    df = pd.DataFrame(data = [], columns = ['V', 'O', 'H', 'L', 'C'], \
                      index = df0["Date"].values)
    cc = list(df.columns)
    df[cc] = df0[cc].values
    TISetUp(myParams.DefinedTechInd, False) # 覆寫全域參數
    ti_graph = TIGraph_Mpf(df, fig_size_x = 24, fig_size_y = 18)
    ti_graph.show_adv("TW2330_Adv_SMA_KD_MACD", ti_main = "SMA", \
                      ti_sub1 = "KD", ti_sub2 = "MACD")
    
    ### 參數資訊輸出
    ShowTIParams() # 顯示在螢幕上
    # 為了得到正常的中文輸出，不使用myLog.Log類別
    Logger = myLog.CreateLogger("TI_main")
    ShowTIParams(Logger) # 寫入日誌檔
    
# Done!
