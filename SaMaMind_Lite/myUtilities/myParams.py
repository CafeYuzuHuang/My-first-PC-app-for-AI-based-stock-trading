# -*- coding: utf-8 -*-

from collections import namedtuple
import os

from myUtilities import myLog

"""
這裡定義了會用到的各種參數
參數命名規則：
    模組名稱使用小駝峰式命名，開頭為"my"
    跨模組使用的變數與函式名稱使用大駝峰式命名
    局部變數：全字母小寫
    其他：
    類別名稱大寫，類別屬性與方法均小寫
    kwargs與字典鍵值全部小寫，單字以'_'串接
"""

# 日誌設定
DefaultRootLogLevel = myLog.LogLevel.INFO
DefaultErrorLogLevel = myLog.LogLevel.ERROR
DefaultFuncLogLevel = myLog.LogLevel.INFO

# Webdrivers路徑：
_modulepath = os.path.dirname(__file__)
_wdpath = os.path.dirname(_modulepath)  + "\\webdrivers"
Chrome_Path = _wdpath + "\\chromedriver.exe"
Phantomjs_Path = _wdpath + "\\phantomjs-2.1.1-windows\\bin\\phantomjs.exe"

# 預設資料夾與檔名
Rawdata_Foldername = "data scraped" # 爬蟲取得之原始資料
PostP_Foldername = "data postprocessed" # 後處理資料
AI_Foldername = "data ai-model" # AI之模型參數存放區
Trade_Foldername = "data ai-trade" # AI運算結果存放區
User_Inf_Folder = "user infiles" # 使用者建立的參數檔
Test_Infile = "Demo.in" # 爬蟲測試.in檔
TW_List = "TW_list.csv" # 上市櫃商品清單

# 檔案輸出參數
PriceDataExt = ".csv" # 股價資料副檔名(檔名為股票或ETF代碼)
Delimiter = ',' # csv分隔符號
# 證交所檔名格式: TWSE-code.csv
# 櫃買中心檔名格式：TPEX-code.csv
# Yahoo Finance: code.TW.csv, code.TWO.csv
# Yahoo Finance (後處理-還原股價): Adj-code.TW.csv, Adj-code.TWO.csv
# 後處理+技術指標計算後：原檔名.xlsx

# 資料後處理會用到的一些資訊
TWSE_Cols = ["日期", "成交股數", "成交金額", "開盤價", \
             "最高價", "最低價", "收盤價", "漲跌價差", "成交筆數"]
TPEX_Cols = ["日期", "成交仟股", "成交仟元", "開盤", "最高", \
             "最低", "收盤", "漲跌", "筆數"]
YF_Cols = ["Date", "Open", "High", "Low", "Close", \
           "Adj Close", "Volume"]

DataSrc = namedtuple("DataSrc", ["Name", "Type", "Owner", "Cols", "File_Ext", "Sep", "DFmt"])
Src_TWSE = DataSrc("TWSE", "HTML", "gov", TWSE_Cols, PriceDataExt, Delimiter, "YYYY-MM-DD")
Src_TPEX = DataSrc("TPEX", "HTML", "gov", TPEX_Cols, PriceDataExt, Delimiter, "YYYY-MM-DD")
Src_YFin = DataSrc("YFin", "HTML", "Yahoo", YF_Cols, ".csv", ',', "YYYY-MM-DD")

Src_Dict = {Src_TWSE.Name: Src_TWSE,
            Src_TPEX.Name: Src_TPEX,
            Src_YFin.Name: Src_YFin}
Src_List = [Src_TWSE.Name, 
            Src_TPEX.Name, 
            Src_YFin.Name]

# 分析運算參數設定
DefinedTechInd = set(["KD", "MACD", "SMA", "VWMA", "STD", "VIX", "CVOL"])
DaysList = [5, 10, 20, 60, 120, 240] # 技術指標計算常用的天期
TradePeriods = DaysList[1:-1] # [10, 20, 60, 120]
OpenDaysPerYear = 240 # 每年開盤日數(約略值)
OpenDaysPerYearEU = 252 # 每年開盤日數(歐洲期權市場)
DaysPerYear = 365
MaxDaySpan = 900 # 最多處理900筆日K資料 
MinDaySpan = 90 # 至少處理90筆日K資料 
MaxNumItems = 100 # 最多處理100檔個股或ETF或期貨指數

# 日投報率限制(漲跌幅限制)
DailyReturnMax = 10.0 # 最大漲幅 = 10%
DailyReturnMin = -10.0 # 最大跌幅 = -10%

# 浮點數運算誤差容許值
Tiny = 1e-4
Tiny2 = 1e-8
Tiny3 = 1e-12
