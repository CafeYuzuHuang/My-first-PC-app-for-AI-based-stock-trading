# -*- coding: utf-8 -*-

import datetime as dt
import os
import numpy as np
import pandas as pd

from myStockScraper import myScraperSettings as mySet
from myUtilities import myParams
from myUtilities import myLog
from myUtilities import myTechIndices as myTI

"""
myPostProcessing所具有的功能：
(1) 資料整併，比對日期，允許將新資料附加在舊資料之後 
註1：僅支援TWSE & TPEX，不支援Yahoo Finance
註2：檔名需相同，待添加與被添加資料的檔案各置於 Raw_Path 和 PostP_Path
註3：需對應股本變動(上市櫃發行/增減資/除權)還原股價
     目前TWSE與TPEX僅處理非還原股價，故還不用考慮此項

(2) 進行資料格式統一、缺失值處理等

(3) 呼叫myTechIndices進行技術指標計算，並輸出到後處理檔案
註1：格式: Date / OHLC / O%H%L%C% / Volume / 
SMA-abc / SMA%-abc / VWMA-abc / VWMA%-abc / 
MACD / std-abc / vol-abc / CVOL，其中 abc 代表三種不同天期
註2：注意計算O%H%L%C%(即今日OHLC相對於昨收的漲跌幅)時須注意除權息日，
TWSE與TPEX會在漲跌上加上"X"記號，Yahoo Finance無標記
目前程式並未處理這點
"""

# Global variables and constants:
LogName = __name__ # 日誌名稱設為模組名稱
RefPath = '' # 參考路徑
if __name__ == "__main__": # executed
    RefPath = os.path.dirname(os.getcwd())
    LogName = None # 當此腳本被執行時為根日誌(不設名稱)
else: # imported
    modulepath = os.path.dirname(__file__)
    RefPath = os.path.dirname(modulepath)
# print(RefPath)
Logger = myLog.Log(myParams.DefaultRootLogLevel, LogName)

Raw_Path = RefPath + '\\' + myParams.Rawdata_Foldername
PostP_Path = RefPath + '\\' + myParams.PostP_Foldername
if not os.path.isdir(PostP_Path):
    os.mkdir(PostP_Path)

Cols_Base = ["V", "O", "H", "L", "C"]
# 價格為日報酬率，量能為昨量比
Cols_Base_R = ["R-V", "R-O", "R-H", "R-L", "R-C"]
KW_R = myTI.KW_R

Cols_KD = ["KD-FK", "KD-K", "KD-D"]
Cols_MACD = ["MACD-DIF", "MACD-DEF", "MACD-Bar"]
Cols_CVOL = ["CVOL-7", "CVOL-14", "CVOL-28"]
Cols_KD_R = [KW_R + s for s in Cols_KD]
Cols_MACD_R = [KW_R + s for s in Cols_MACD]
Cols_CVOL_R = [KW_R + s for s in Cols_CVOL]

# 以下全域變數將被函數修改
Cols_R_Included = Cols_Base + Cols_Base_R # default columns
Cols_Selected = Cols_Base # default columns

TradePeriod = myParams.TradePeriods[1] # 20 days
ABC = sorted(set([myParams.DaysList[1], 
                  myParams.DaysList[2], 
                  myParams.DaysList[3]])) # 10, 20, 60; 不重複
SelectedTechInd = set() # default: no tech indicators applied
Cols_SMA = ["SMA-" + str(s) for s in ABC]
Cols_VWMA = ["VWMA-" + str(s) for s in ABC]
Cols_STD = ["STD-" + str(s) for s in ABC]
Cols_VIX = ["VIX-" + str(s) for s in ABC]
Cols_SMA_R = [KW_R + s for s in Cols_SMA]
Cols_VWMA_R = [KW_R + s for s in Cols_VWMA]
Cols_STD_R = [KW_R + s for s in Cols_STD]
Cols_VIX_R = [KW_R + s for s in Cols_VIX]


@myLog.Log_Error(myParams.DefaultErrorLogLevel)
# @myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def UpdateColNames():
    """ 對應更新技術指標欄位標籤 """
    global Cols_SMA, Cols_VWMA, Cols_STD, Cols_VIX
    global Cols_SMA_R, Cols_VWMA_R, Cols_STD_R, Cols_VIX_R
    Cols_SMA = ["SMA-" + str(s) for s in ABC]
    Cols_VWMA = ["VWMA-" + str(s) for s in ABC]
    Cols_STD = ["STD-" + str(s) for s in ABC]
    Cols_VIX = ["VIX-" + str(s) for s in ABC]
    Cols_SMA_R = [KW_R + s for s in Cols_SMA]
    Cols_VWMA_R = [KW_R + s for s in Cols_VWMA]
    Cols_STD_R = [KW_R + s for s in Cols_STD]
    Cols_VIX_R = [KW_R + s for s in Cols_VIX]
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
# @myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def SetTradeDays(day = 20):
    """
    設定交易週期與技術指標天期
    交易週期(TradePeriod)是"期望"的持倉天期，不一定與實際情況相當
    """
    global TradePeriod, ABC
    xx = min(enumerate(myParams.TradePeriods), \
             key=lambda d: abs(day - d[1])) # xx = (index, value)
    if day != xx[1]:
        Logger.info("Specified trade day period %d will be modifed." % day)
    x = xx[0] # tick for the trade day period
    TradePeriod = myParams.TradePeriods[x]
    ABC = sorted(set([myParams.DaysList[x], 
                      myParams.DaysList[x + 1], 
                      myParams.DaysList[x + 2]]))
    Logger.info("Trade day period: %d" % TradePeriod)
    Logger.info("Day periods for tech-indicators: %d, %d, %d" \
                % (ABC[0], ABC[1], ABC[2]))
    UpdateColNames()
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
# @myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def GetSelectedColNames(kwset):
    """ 取得選取技術指標後，全部資料欄位名稱 """
    global Cols_Selected, Cols_R_Included
    # 儘管這裡沒將 Cols_Base 宣告為 global
    # 使用 cols = Cols_Base，之後 cols 的變動將會對應到 Cols_Base
    # 因此必須使用 deep copy，避免兩變數指向相同記憶體位址...
    cols = Cols_Base.copy()
    cols_r = Cols_Base_R.copy()
    if len(kwset) == 0:
        Logger.info("Empty keyword set is found.")
        Cols_Selected = cols
        Cols_R_Included = cols + cols_r
        return None
    for kw in kwset:
        if kw.upper() == "KD":
            cols += Cols_KD
            cols_r += Cols_KD_R
        elif kw.upper() == "MACD":
            cols += Cols_MACD
            cols_r += Cols_MACD_R
        elif kw.upper() == "SMA":
            cols += Cols_SMA
            cols_r += Cols_SMA_R
        elif kw.upper() == "VWMA":
            cols += Cols_VWMA
            cols_r += Cols_VWMA_R
        elif kw.upper() == "STD":
            cols += Cols_STD
            cols_r += Cols_STD_R
        elif kw.upper() == "VIX":
            cols += Cols_VIX
            cols_r += Cols_VIX_R
        elif kw.upper() == "CVOL":
            cols += Cols_CVOL
            cols_r += Cols_CVOL_R
    Cols_Selected = cols.copy()
    Cols_R_Included = (cols + cols_r).copy()
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
# @myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def SetSelectedTechInd(kwlist):
    """ 提供技術指標清單，確認各項目是否已被程式定義 """
    selectedtechind = set()
    if kwlist == [] or kwlist is None:
        Logger.info("Empty keyword list is found.")
        return selectedtechind
    for kw in kwlist:
        if kw in myParams.DefinedTechInd:
            selectedtechind.add(kw)
        else:
            Logger.info("Tech indicator keyword %s is undefined." % kw)
    return selectedtechind

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def GetSelectedTechInd(td, kwlist):
    """ 
    根據使用者輸入的操作天期和技術指標清單做設定
    並依此更新後處理資料的總欄位
    """
    global SelectedTechInd
    SetTradeDays(td)
    SelectedTechInd = SetSelectedTechInd(kwlist)
    GetSelectedColNames(SelectedTechInd)
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def GetRawList(dpath, datasrc):
    """ 根據data source類型，取得檔案清單 """
    flist_raw = os.listdir(dpath) # candidates
    flist = []
    for ff in flist_raw:
        ffext = os.path.splitext(ff)[-1]
        ffname = os.path.splitext(ff)[0]
        if ffext.lower() == datasrc.File_Ext.lower():
            if datasrc.Name == "YFin" or datasrc.Owner == "Yahoo":
                if mySet.TWCodeCheck(ffname.strip(".TW")):
                    flist.append(ff)
                elif mySet.TWCodeCheck(ffname.strip(".TWO")):
                    flist.append(ff)
            elif datasrc.Name == "TWSE":
                if mySet.TWCodeCheck(ffname.strip("TWSE-")):
                    flist.append(ff)
            elif datasrc.Name == "TPEX":
                if mySet.TWCodeCheck(ffname.strip("TPEX-")):
                    flist.append(ff)
            else:
                if mySet.TWCodeCheck(ffname):
                    flist.append(ff)
    Logger.info("# of valid files: %d" % len(flist))
    invalids = len(flist_raw) - len(flist)
    Logger.info("# of invalid files and dirs: %d" % invalids)
    return flist

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
# @myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def APDFmtConv(ap_dfmt):
    """ 將日期格式 YYYY/MM/DD 轉換成一般格式 YYYY-MM-DD """
    return mySet.CorrectIsoformat(ap_dfmt, '/')

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
# @myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def PriceConv(tw_p):
    """
    針對TWSE與TPEX非數值價位做處理
    例如，停牌時該各股當日成交量 = 0，四價位 = '--'
    """
    if tw_p == '--' or tw_p == '-': # 當日無成交價格資訊
        return np.nan
    else: # 當日正常交易，避免價位格式為帶','之字串格式
        return float(str(tw_p).replace(',', ''))

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def GetVOHLCAdj(fpath, datasrc):
    """
    讀入檔案，然後取出成交量和四價位數據，日期則作為索引
    日期格式YYYY-MM-DD，價位單位NTD/股，成交量單位為股(=1/1000張)
    僅允許Yahoo Finance資料源使用
    """
    df_new = pd.DataFrame(data = [])
    if datasrc.Name == myParams.Src_List[2]: # Yahoo finance
        df = pd.read_csv(fpath, delimiter = datasrc.Sep, header = 0, \
                         index_col = None, skiprows = None)
        cols = datasrc.Cols
        diff = df[cols[5]].apply(float) - df[cols[4]].apply(float)
        sd = df[cols[0]] # Already formatted after scraping
        sd = pd.to_datetime(sd)
        so = df[cols[1]].apply(float) + diff # all diff <= 0
        sh = df[cols[2]].apply(float) + diff
        sl = df[cols[3]].apply(float) + diff
        sc = df[cols[5]].apply(float) # Adjusted close price
        sv = df[cols[6]]
        df_new = pd.concat([sv, so, sh, sl, sc], axis = 1)
        df_new.index = sd
    else:
        Logger.warning("Data source %s is not supported!" % datasrc.Name)
    df_new.columns = Cols_Base # 統一更改欄位名稱
    df_new.index.name = "Date"
    return df_new

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def GetVOHLC(fpath, datasrc):
    """
    讀入檔案，然後取出成交量和四價位數據，日期則作為索引
    日期格式YYYY-MM-DD，價位單位NTD/股，成交量單位為股(=1/1000張)
    根據提供的data source決定處理資料的方法
    """
    df_new = pd.DataFrame(data = [])
    if datasrc.Name == myParams.Src_List[0]: # TWSE
        df = pd.read_csv(fpath, delimiter = datasrc.Sep, header = 0, \
                         index_col = None, skiprows = None)
        cols = datasrc.Cols
        sd = df[cols[0]] # Already formatted after scraping
        sd = pd.to_datetime(sd)
        so = df[cols[3]].apply(PriceConv)
        sh = df[cols[4]].apply(PriceConv)
        sl = df[cols[5]].apply(PriceConv)
        sc = df[cols[6]].apply(PriceConv)
        sv = df[cols[1]].apply(lambda x: int(str(x).replace(',', '')))
        df_new = pd.concat([sv, so, sh, sl, sc], axis = 1)
        df_new.index = sd
    elif datasrc.Name == myParams.Src_List[1]: # TPEX
        df = pd.read_csv(fpath, delimiter = datasrc.Sep, header = 0, \
                         index_col = None, skiprows = None)
        cols = datasrc.Cols
        sd = df[cols[0]] # Already formatted after scraping
        sd = pd.to_datetime(sd)
        so = df[cols[3]].apply(PriceConv)
        sh = df[cols[4]].apply(PriceConv)
        sl = df[cols[5]].apply(PriceConv)
        sc = df[cols[6]].apply(PriceConv)
        sv = df[cols[1]].apply(lambda x: float(str(x).replace(',', ''))*1000)
        df_new = pd.concat([sv, so, sh, sl, sc], axis = 1)
        df_new.index = sd
    elif datasrc.Name == myParams.Src_List[2]: # Yahoo finance
        df = pd.read_csv(fpath, delimiter = datasrc.Sep, header = 0, \
                         index_col = None, skiprows = None)
        cols = datasrc.Cols
        sd = df[cols[0]] # Already formatted after scraping
        sd = pd.to_datetime(sd)
        so = df[cols[1]].apply(float)
        sh = df[cols[2]].apply(float)
        sl = df[cols[3]].apply(float)
        sc = df[cols[4]].apply(float)
        sv = df[cols[6]]
        df_new = pd.concat([sv, so, sh, sl, sc], axis = 1)
        df_new.index = sd
    else:
        Logger.warning("Undefined datasrc name!")
    df_new.columns = Cols_Base # 統一更改欄位名稱
    df_new.index.name = "Date"
    return df_new

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
# @myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def ShowDropedRows(old, new, kw):
    """ GetCleanData 的輔助函式 """
    diff = old - new
    if diff > 0:
        Logger.info("Drop %d rows by finding any %s." % (diff, kw))
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def GetCleanData(data):
    """
    清除任一欄位帶有缺失值的資料，一旦找到便將該列刪除
    目前考量的缺失值包含： NaN, '', null, '-', 0等
    為避免拋出例外加上用於分析髒資料的特徵，故分步驟依序清理
    """
    data_rows = data.shape[0]
    # (1) 排除np.NaN，注意其為numeric type，np.isreal = True
    data_tmp = data.dropna(axis = 0, inplace = False)
    tmp_rows = data_tmp.shape[0]
    ShowDropedRows(data_rows, tmp_rows, "NaN")
    # (2) 排除非數值(包含空白)，保留各欄位均為數值的資料
    rowisnum = data_tmp.apply(np.isreal).all(axis = 1)
    data_new = data_tmp.loc[rowisnum, :]
    new_rows = data_new.shape[0]
    ShowDropedRows(tmp_rows, new_rows, "non-numeric")
    # (3) 保留正值，不包含0
    rowispos = (data_new > 0).all(axis = 1)
    df = data_new.loc[rowispos, :]
    ShowDropedRows(new_rows, df.shape[0], "non-positive")
    return df

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def AppendCleanRawData(datasrc, dpath, flist):
    """
    迴圈處理個別檔案，
    依序進行格式統一、資料清理，以及與資料庫數據合併
    """
    for ff in flist:
        fpath = dpath + '\\' + ff # 原始數據檔案的完整路徑
        spath = PostP_Path + '\\' + ff # 後處理目標路徑+檔名
        data = GetVOHLC(fpath, datasrc) # 取得格式統一後的基本欄位資料
        data = GetCleanData(data) # 刪除帶有缺失值的日期欄位數據
        if datasrc.Name == myParams.Src_List[2]:
            # Yahoo Finance的還原股價數據
            # 因為還原股價是以今日倒推回去，因此舊數據需要不斷被更新
            # 為求簡潔，就不做資料整併同時還要修正還原股價...
            data_adj = GetVOHLCAdj(fpath, datasrc)
            data_adj = GetCleanData(data_adj)
            spath2 = PostP_Path + "\\Adj-" + ff
            data_adj.to_csv(spath2, sep = myParams.Delimiter, \
                            header = True, index = True)
        # 整併資料，並剔除重複者，這部分只支援TWSE & TPEX
        if datasrc.Name == myParams.Src_List[0] or \
            datasrc.Name == myParams.Src_List[1]:
            if os.path.isfile(spath): # file exists
                df = pd.read_csv(spath, delimiter = myParams.Delimiter, \
                                 header = 0, index_col = 0, \
                                 skiprows = None, parse_dates = True)
                his_data = df.iloc[:, :len(Cols_Base)]
                # 簡單確認舊有數據最晚天期在新抓數據最早天期的後面
                # 表示新抓數據與舊有數據之間有重疊，可以推論中間沒有漏數據
                if data.index[0] < his_data.index[-1]: # overlapped
                    data_new = pd.concat([his_data, data], axis = 0)
                    data_new.drop_duplicates(inplace = True, keep = 'last')
                    data_new.to_csv(spath, sep = myParams.Delimiter, \
                                    header = True, index = True)
                else:
                    Logger.warning("No overlapping between %s and %s!" % \
                                   (fpath, spath))
                    Logger.warning("New data is not appended yet.")
            else:
                Logger.info("Get a new post-processed one: %s" % fpath)
                data.to_csv(spath, sep = myParams.Delimiter, \
                            header = True, index = True)
        else:
            # 對於非證交所與櫃買中心資料，就不費心做資料整併而直接覆寫！
            data.to_csv(spath, sep = myParams.Delimiter, \
                        header = True, index = True)
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
# @myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def GetTechIndData(df):
    """ 技術指標計算 """
    kd = macd = sma = vwma = std = vix = df_r = pd.DataFrame(data = [])
    kd_r = macd_r = sma_r = vwma_r = std_r = vix_r = df_r
    kd = myTI.KD(df, Cols_KD)
    macd = myTI.MACD(df, Cols_MACD)
    sma = myTI.SMA(df, Cols_SMA, ABC)
    vwma = myTI.VWMA(df, Cols_VWMA, ABC)
    std = myTI.STD(df, Cols_STD, ABC)
    vix = myTI.VIX(df, Cols_VIX, ABC)
    cvol = myTI.CVOL(df, Cols_CVOL)
    df_r = myTI.PriceDailyReturn(df, Cols_Base_R, True) # 必算
    kd_r = myTI.TI_DailyReturn(kd, Cols_KD_R)
    macd_r = myTI.TI_DailyReturn(macd, Cols_MACD_R)
    sma_r = myTI.TI_DailyReturn(sma, Cols_SMA_R)
    vwma_r = myTI.TI_DailyReturn(vwma, Cols_VWMA_R)
    std_r = myTI.TI_DailyReturn(std, Cols_STD_R)
    vix_r = myTI.TI_DailyReturn(vix, Cols_VIX_R)
    cvol_r = myTI.TI_DailyReturn(cvol, Cols_CVOL_R)
    df_ti = pd.concat([df, kd, macd, sma, vwma, std, vix, cvol, \
                       df_r, kd_r, macd_r, sma_r, vwma_r, std_r, \
                       vix_r, cvol_r], axis = 1)
    return df_ti.round(4)

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def AppendTechIndData(flist, isdr = True):
    """ 附加技術指標計算結果至 DataFrame 並輸出存檔 """
    techind = list(SelectedTechInd)
    if len(techind) == 0:
        Logger.info("Empty tech indicator list is found!")
    myTI.TISetUp(techind, isdr) # Initialization
    for ff in flist:
        ppath = PostP_Path + '\\' + ff
        tpath = PostP_Path + '\\' + os.path.splitext(ff)[0] + ".xlsx"
        df = pd.read_csv(ppath, delimiter = myParams.Delimiter, \
                         header = 0, index_col = 0, skiprows = None, \
                         parse_dates = True)
        df.index = pd.to_datetime(df.index)
        df_ti = GetTechIndData(df) # 技術指標計算
        # 換一個副檔名，除了避免覆蓋原本檔案外用.xlsx也比較便於立即用excel繪圖
        df_ti.to_excel(tpath, header = True, index = True)
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def Main(td, kwlist, datasrc):
    """
    主函式：進行以下各項數據後處理
    datasrc必須為DataSrc類型
    當datasrc.Name選擇TWSE或TPEX，結果相同不會有差異
    """
    if not os.path.isdir(PostP_Path):
        os.mkdir(PostP_Path)
    
    # 選取需要的技術指標與操作天期設定
    # td = myParams.TradePeriods[1]
    # kwlist = list(myParams.DefinedTechInd)
    GetSelectedTechInd(td, kwlist)
    
    # 引入參數型態驗證：
    if str.find(repr(type(datasrc)), "DataSrc") == -1: # not found!
        Logger.warning("Invalid type %s in arg." % repr(type(datasrc)))
        Logger.warning("Now exit ...")
        return None
    Logger.info("Show datasrc name: %s, type: %s, owner: %s" % \
                (datasrc.Name, datasrc.Type, datasrc.Owner))
    # 初始化並取得rawdata檔案清單
    # DataSrc.Name / Type / Owner / File_Ext 將會用於條件判斷
    dpath = Raw_Path
    flist = []
    if datasrc.Type == "HTML": # 利用爬蟲擷取存下的檔案
        flist = GetRawList(dpath, datasrc)
    else:
        Logger.warning("Undefined datasrc.type found! Now exit ...")
        return None
    # 進入迴圈處理個別檔案
    # DataSrc.Name / Cols / Sep / DFmt 將會用於條件判斷
    AppendCleanRawData(datasrc, dpath, flist)
    # 進行技術指標計算
    AppendTechIndData(flist,)
    return None

# Function testing 
if __name__ == "__main__":
    t_start = dt.datetime.now()
    print("***** Module myPostProcessing test: *****")
    SrcDict = myParams.Src_Dict
    SrcList = myParams.Src_List
    
    # td = myParams.TradePeriods[1]
    # kwlist = list(myParams.DefinedTechInd)
    td = 7
    kwlist = ["MACD", "SMA", "CVOL"]
    
    # Main(td, kwlist, SrcDict[SrcList[2]]) # Yahoo finance
    Main(td, kwlist, SrcDict[SrcList[0]]) # TWSE
    Main(td, kwlist, SrcDict[SrcList[1]]) # TPEX
    t_end = dt.datetime.now()
    print("Total ellapsed time is: ", t_end - t_start)
# Done!
