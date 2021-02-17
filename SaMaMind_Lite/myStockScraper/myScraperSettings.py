# -*- coding: utf-8 -*-

import random as rd
from collections import namedtuple
import datetime as dt
import math

from myUtilities import myParams

"""
myScraperSettings.py 的功能：
這裡定義了爬蟲會用到的網路相關設定(需定期維護以下資訊)
"""

# 抓取歷史股價資訊的來源：google finance 日後視需求進行之
PriceDataSrc = ("TWSE_TPEX", "Yahoo_Finance")

# 擷取資料之欄位名稱：
YF_Cols = myParams.YF_Cols
TWSE_Cols = myParams.TWSE_Cols
TPEX_Cols = myParams.TPEX_Cols

"""
定義股票與ETF清單列資料的名稱如下
目前需要抓取的清單有：
上市股票、上市ETF、上櫃股票、上櫃ETF (共四項)

全部項目：
Rows = namedtuple('Row', ['ID', 'ISIN', 'TW_Code', 'Name', 'Market', \
                          'Sec_Type', 'Field', 'Start_Date', 'CFI', 'Remark'])
"""
# 只取用以下項目：未來可使用x.TW_Code, x.Name等方式取得資料
Rows = namedtuple("Row", ["TW_Code", "Name", "Market", "Sec_Type"])

def GetRows(row):
    """
    將網頁擷取下來的每筆資料取出所需的欄位後，建立namedtuple格式
    twcode = row[2]
    name = row[3]
    market = row[4]
    sectype = row[5]
    return mySet.Rows(twcode, name, market, sectype)
    """
    return Rows(str(row[2]), str(row[3]), str(row[4]), str(row[5]))

# *** 輔助功能 ***
# 日期格式檢查與轉換
def CorrectIsoformat(dfmt_old, cha = '-'):
    """ input: "yyyy-dd-mm", "yyyy-d-m", "yyyy/dd/mm", etc. """
    ymd = dfmt_old.split(cha)
    if len(ymd[1]) == 1:
        ymd[1] = '0' + ymd[1]
    if len(ymd[2]) == 1:
        ymd[2] = '0' + ymd[2]
    dfmt_new = ymd[0] + '-' + ymd[1] + '-' + ymd[2] 
    return dfmt_new

def DateFmtConv(tw_dfmt):
    """ 將台灣日期格式 yyy/MM/DD 轉換成一般格式 YYYY-MM-DD """
    ymd = tw_dfmt.split('/')
    ymd[0] = str(int(ymd[0]) + 1911)
    if len(ymd[1]) == 1:
        ymd[1] = '0' + ymd[1]
    if len(ymd[2]) == 1:
        ymd[2] = '0' + ymd[2]
    dfmt = ymd[0] + '-' + ymd[1] + '-' + ymd[2] 
    return dfmt

# 台灣證券代碼檢查
def IsETFCode(code):
    """
    ETF編號檢查，做以下判斷：
    (1) 開頭需帶有兩個0，例如0050, 00692, 006203
    (2) 字串總長度4以上 (目前ETF編碼字串長4至6)
    (3) 只有最後一個字元可以是字母，前面必須都是數字0-9
    結尾不辨別，有些會帶有字母表示幣別或正反向槓桿等
    如00677U, 00637L, 00632R
    (4) 確認沒有奇怪的符號混進來，尤其字尾容易出現'\n'或'\t'
    """
    try:
        if str(code[0]) != '0' or str(code[1]) != '0':
            raise Exception("Code not start with '00'. See rule-1.")
        elif len(code) < 4 or len(code) > 7:
            raise Exception("Invalid code length. See rule-2.")
        elif not all(str(i).isdigit() for i in code[:-1]):
            raise Exception("Invalid code format. See rule-3.")
        elif not str(code).isalnum(): # exists character(s) not 0-9a-zA-Z
            raise Exception("Exist symbols (not 0-9a-zA-Z). See rule-4.")
        return True
    except Exception:
        pass
    except:
        print("Unexpected exception found in IsETFCode()...")
        pass
    return False

def IsStockCode(code):
    """ 檢查是否為台灣上市櫃股票代碼(介於1000-9999) """
    try:
        int(code) # to raise value error (not a digit)
        if int(code) < 1000 or int(code) > 9999:
            raise Exception("Invalid code range.")
        return True
    except ValueError:
        pass
    except Exception:
        pass
    except:
        print("Unexpected exception found in IsStockCode()...")
        pass
    return False

def TWCodeCheck(code):
    """ 檢查是否為台灣上市櫃股票或ETF代碼 """
    if IsETFCode(code) or IsStockCode(code):
        return True
    else: # undefined or invalid
        return False

# 請求狀態確認
def CodeIdentify(http_status_code):
    """ Http Status Code classification """
    if int(http_status_code) - 500 >= 0:
        status_msg = 'Server errors'
    elif int(http_status_code) - 400 >= 0:
        status_msg = 'Client errors'
    elif int(http_status_code) - 300 >= 0:
        status_msg = 'Redirection'
    elif int(http_status_code) - 200 >= 0:
        # 200 is the expected status code!
        status_msg = 'Success'
    elif int(http_status_code) - 100 >= 0:
        status_msg = 'Informational response'
    else:
        status_msg = 'Unidentified'        
    return status_msg + ' - ' + str(http_status_code)

# 設定request間隔時間 
ts = 5 # 睡眠秒數，用於time.sleep函式

def RandTS(ts_a = 4, ts_b = 8):
    """ 設定隨機間隔時間，避免爬蟲時request過於頻繁 """
    rts = ts_a + rd.randint(a=ts_a, b=ts_b)*rd.random()
    return rts

# *** Url 與 Headers ***
# Yahoo finance 設定
def GenerateYFUrl(code_ex):
    """
    產生對應的url，頁面位置在該個股或ETF的歷史資料頁面
    預設顯示的是一年內的日K資訊
    如果要再調整時間範圍的話必須透過webdriver做網頁按鍵操作
    參考：https://github.com/vb100/Yahoo-Finance-Stock-Scrapper-Selenium/blob/master/yahoo_stock_scrapper_selenium.py
    """
    # url範例：
    # https://finance.yahoo.com/quote/5234.TW/history?p=5234.TW
    # 股票名稱結尾：美股無，台股上市為.TW, 上櫃為.TWO
    base = "https://finance.yahoo.com/quote/"
    url = base + code_ex + "/history?p=" + code_ex
    return url

def GenerateYFUrlTP(code_ex, sd, ed):
    """
    產生對應的url，頁面位置在該個股或ETF的歷史資料頁面
    時間週期透過url指定，載入後的頁面可以取得原始碼後再用bs4分析
    和透過selenium操作按鍵相比，是偷吃步的做法
    """
    sd_new = CorrectIsoformat(sd, '-')
    ed_new = CorrectIsoformat(ed, '-')
    # 產生時間戳
    sddt = dt.datetime.strptime(sd_new, '%Y-%m-%d')
    eddt = dt.datetime.strptime(ed_new, '%Y-%m-%d')
    tp1 = math.trunc(dt.datetime.timestamp(sddt))
    tp2 = math.trunc(dt.datetime.timestamp(eddt))
    
    # url範例：當設定好Time Period或Show或Frequency，按下Apply按鈕後產生如下的url
    # https://finance.yahoo.com/quote/5234.TW/history?period1=1262304000&period2=1603584000&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
    # 股票名稱結尾：美股無，台股上市為.TW, 上櫃為.TWO
    base = "https://finance.yahoo.com/quote/"
    additional = \
        "&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true"
    url = base + code_ex + "/history?period1=" + str(tp1) + \
        "&period2=" + str(tp2)
    myurl = url + additional
    return myurl

# TWSE & TPEX 抓取盤後資料設定
def GenerateTWSEUrls(code, sd, ed):
    """
    產生對應的url清單，每個url對應一個月的資訊；上市ETF可通用
    頁面原始碼以bs4或其他模組分析擷取資料
    """
    # Url範例 (日期不重要可忽略)
    # https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=html&date=20201026&stockNo=2330
    base = "https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=html&date="
    midstr = "01&stockNo=" # 日期設為第一天
    sddt = dt.datetime.fromisoformat(CorrectIsoformat(sd, '-'))
    eddt = dt.datetime.fromisoformat(CorrectIsoformat(ed, '-'))
    nourls = (eddt.month - sddt.month + 1) + 12*(eddt.year - sddt.year)
    cur_m = sddt.month
    cur_y = sddt.year
    urllist = []
    for i in range(0, nourls):
        str_m = '0' + str(cur_m) if len(str(cur_m)) == 1 else str(cur_m)
        tmp = base + str(cur_y) + str_m + midstr + code
        urllist.append(tmp)
        cur_m += 1
        if cur_m > 12:
            cur_y += 1
            cur_m = 1
    return urllist

def GenerateTPEXUrls(code, sd, ed):
    """
    產生對應的url清單，每個url對應一個月的資訊；上櫃ETF可通用
    頁面原始碼以bs4或其他模組分析擷取資料
    """
    # Url範例：
    # https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_print.php?l=zh-tw&d=109/04&stkno=006201&s=0,asc,0
    base = "https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_print.php?l=zh-tw&d="
    midstr = "&stkno="
    endstr = "&s=0,asc,0"
    sddt = dt.datetime.fromisoformat(CorrectIsoformat(sd, '-'))
    eddt = dt.datetime.fromisoformat(CorrectIsoformat(ed, '-'))
    nourls = (eddt.month - sddt.month + 1) + 12*(eddt.year - sddt.year)
    cur_m = sddt.month
    cur_y = sddt.year - 1911 # 民國年
    urllist = []
    for i in range(0, nourls):
        str_m = '0' + str(cur_m) if len(str(cur_m)) == 1 else str(cur_m)
        tmp = base + str(cur_y) + '/' + str_m + midstr + code + endstr
        urllist.append(tmp)
        cur_m += 1
        if cur_m > 12:
            cur_y += 1
            cur_m = 1
    return urllist


"""
以下僅url與referer不同，其他項目相同
url中注意market與issuetype對應的值無規則
referer不一定每次都相同，例如issuetype常常不同(無明顯規則)
"""
TWSE_Stock_Url = "https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=1&issuetype=1&industry_code=&Page=1&chklike=Y"
TPEX_Stock_Url = "https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=2&issuetype=4&industry_code=&Page=1&chklike=Y"
TWSE_ETF_Url = "https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=1&issuetype=I&industry_code=&Page=1&chklike=Y"
TPEX_ETF_Url = "https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=2&issuetype=3&industry_code=&Page=1&chklike=Y"

"""
若以下請求表頭出現問題，可改用OrderedDict
TWSE_Stock_Headers = \
OrderedDict([("Accept", "text/html..."), (..., ...), ...])
在Python3.7以上的版本dict本身自帶插入順序，故應無此需求
"""
TWSE_Stock_Headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7,ja;q=0.6",
    "Connection": "keep-alive",
    "Host": "isin.twse.com.tw",
    "Referer": "https://isin.twse.com.tw/isin/class_i.jsp?kind=2&owncode=&stockname=&isincode=&markettype=1&issuetype=3&industry_code=",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36"}

TWSE_ETF_Headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7,ja;q=0.6",
    "Connection": "keep-alive",
    "Host": "isin.twse.com.tw",
    "Referer": "https://isin.twse.com.tw/isin/class_i.jsp?kind=2&owncode=&stockname=&isincode=&markettype=1&issuetype=3&industry_code=",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36"}

TPEX_Stock_Headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7,ja;q=0.6",
    "Connection": "keep-alive",
    "Host": "isin.twse.com.tw",
    "Referer": "https://isin.twse.com.tw/isin/class_i.jsp?kind=2&owncode=&stockname=&isincode=&markettype=2&issuetype=1&industry_code=",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36"}

TPEX_ETF_Headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7,ja;q=0.6",
    "Connection": "keep-alive",
    "Host": "isin.twse.com.tw",
    "Referer": "https://isin.twse.com.tw/isin/class_i.jsp?kind=2&owncode=&stockname=&isincode=&markettype=2&issuetype=1&industry_code=",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36"}

TWSEPrice_Headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7,ja;q=0.6",
    "Connection": "keep-alive",
    "Host": "www.twse.com.tw",
    "Referer": "https://www.twse.com.tw/zh/page/trading/exchange/STOCK_DAY.html",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36"}

TPEXPrice_Headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7,ja;q=0.6",
    "Connection": "keep-alive",
    "Host": "www.tpex.org.tw",
    "Referer": "https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43.php?l=zh-tw",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36"}

# Done!
