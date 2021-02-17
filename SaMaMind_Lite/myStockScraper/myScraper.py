# -*- coding: utf-8 -*-

from time import sleep
import datetime as dt
import os
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import pandas as pd

from myStockScraper import myScraperSettings as mySet
from myUtilities import myParams
from myUtilities import myLog

"""
myScraper.py目前具有的功能如下：
(1) 從證交所獲取股票與ETF清單，建立Item_Dict
(2) 讀取使用者的觀察池清單，建立User_Pool(資料爬蟲對象)
(3) 依使用者設定的時間範圍，從指定網站爬取歷史交易資訊
註：Yahoo Finance雖便於爬蟲(可一次把全範圍歷史數據抓下來)，但
缺失資料較多(尤其ETF)，且成交量數值與證交所和櫃買中心不符(沒算到盤後零股?)
因此較建議抓取證交所和櫃買中心數據，為信賴度考量
註2：後處理由myPostProcessing進行
註3：由於後處理功能會做資料整併，而抓取證交所與櫃買中心數據時都是以月為單位下載資料
     故不在爬蟲階段時去剃除多餘資料，而是留到資料整併時再來處理
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

Raw_Path = RefPath + '\\' + myParams.Rawdata_Foldername
if not os.path.isdir(Raw_Path):
    os.mkdir(Raw_Path)
TW_List_File = RefPath + '\\' + myParams.TW_List

Logger = myLog.Log(myParams.DefaultRootLogLevel, LogName) # Set logger

# 以下全域變數將被函數修改
Save_Counter = 0 # 計算打開清單檔存入的次數(最多四次)
Item_Dict = {} # 上市櫃股票與ETF清單(以代碼搜尋)
User_Pool = [] # 使用者的觀察池清單


@myLog.Log_Error(myParams.DefaultErrorLogLevel)
# @myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def SaveItemList(dict_rows, csvf, csvdel = ','):
    """ 將個股與ETF列表儲存下來 """
    fmod = 'a' # append mode, write only (a+ for readable)
    writecol = False
    if Save_Counter == 0:
        fmod = 'w' # overwrite mode, write only (w+ for readable)
        writecol = True # export columns names as header
    data = []
    cols = [item for item in mySet.Rows._fields]
    for k in dict_rows.keys():
        data.append([dict_rows[k][i] for i in range(0, len(dict_rows[k]))])
    df = pd.DataFrame(data = data, index = None, columns = cols)
    df.to_csv(csvf, sep = csvdel, mode = fmod, encoding = "utf-8", 
              index = False, header = writecol)
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
# @myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def LoadItemList(csvf, csvdel = ','):
    """ 讀取個股與ETF列表 """
    global Item_Dict
    df = pd.read_csv(csvf, sep = csvdel, header = 0, \
                     index_col = None, skiprows = None)
    for i in range(0, df.shape[0]):
        row = mySet.Rows._make(df.iloc[i, :])
        Item_Dict[row.TW_Code] = row # Add to dict
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def FetchItemList(url, headers = None):
    """ 從證交所ISIN網頁抓取商品清單 """
    try:
        session = requests.Session()
        if headers == {} or headers is None:
            Logger.warning("Request headers is not specified!")
            req = session.get(url)
        else:
            req = session.get(url, headers = headers)
        msg = mySet.CodeIdentify(req.status_code)
        Logger.debug("Status code validation:")
        # 手動拋出例外
        req.raise_for_status() # raise if status_code >= 400 
        if req.status_code != 200: # 非常嚴格的狀態碼限定
            Logger.warning("Request status code must be 200 (O.K.)")
            # raise Exception("Request status code must be 200 (O.K.)")
    except requests.exceptions.HTTPError as e:
        Logger.error("HTTP Error: %s" % e)
    except requests.exceptions.ConnectionError as e:
        Logger.error("Connection Error: %s" % e)
    except requests.exceptions.Timeout as e:
        Logger.error("Timeout Error: %s" % e)
    except requests.exceptions.RequestException as e:
        Logger.error("Ambiguous exception occurred! %s" % e.response.text)
        Logger.error("Status code and message: %s" % msg)
    except Exception as e:
        Logger.critical("Unknown error occurred! Show exception message:")
        Logger.critical("{}".format(e))
    else:
        # 對於格式化HTML表格，也可以使用Pandas來處理
        Logger.debug("Now starts to make the soup.")
        # 編碼變更：有需要再用
        # req.encoding = 'utf-8'
        # req.encoding = 'big5'
        soup = BeautifulSoup(req.text, "html.parser") # default
        # soup = BeautifulSoup(req.text, "lxml") # recommended?
        data = {}
        invaliddata = []
        for row in soup.find("table", {"class": "h4"}).find_all('tr'):
            # 將文字串的空白與換行符號洗掉，並轉換成list格式
            tmp = row.get_text().strip().split('\n')
            tmp = [a.strip() for a in tmp]
            tmp = mySet.GetRows(tmp)
            if mySet.TWCodeCheck(tmp.TW_Code) == True: # Valid code
                data[tmp.TW_Code] = tmp
            else: # 收集不正確的資料(optional)
                # invaliddata.append(tmp)
                invaliddata.append(row.get_text())
        # 處理不正確的資料(optional)
        Logger.debug("Now starts to check invalid data:")
        if len(invaliddata) > 0:
            Logger.warning("FetchItemList found invalid data!")
            outf = RefPath + "\\FetchItemList.err"
            fmod = 'a'
            if Save_Counter == 0:
                fmod = 'w'
            with open(outf, fmod, encoding = "utf-8") as f:
                f.writelines("%s\n" % item for item in invaliddata)
        return data
    return None


@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def UpdateItemList(marketid):
    """ 從證交所收集最新的上市個股與ETF列表 """
    # Get info
    if marketid == 1: # 上市
        stock_headers = mySet.TWSE_Stock_Headers
        stock_url = mySet.TWSE_Stock_Url
        etf_headers = mySet.TWSE_ETF_Headers
        etf_url = mySet.TWSE_ETF_Url
    elif marketid == 2: # 上櫃
        stock_headers = mySet.TPEX_Stock_Headers
        stock_url = mySet.TPEX_Stock_Url
        etf_headers = mySet.TPEX_ETF_Headers
        etf_url = mySet.TPEX_ETF_Url
    else:
        Logger.warning("Unknown market ID: %s. Exit directly!" % marketid)
        return None
    
    global Save_Counter
    s = {0: "stock", 1: "ETF"} # for logging
    m = {1: "TWSE", 2: "TPEX"} # for logging
    headers = [stock_headers, etf_headers]
    urls = [stock_url, etf_url]
    for i in range(0, len(urls)):
        # Fetch data
        Logger.info("Now fetch %s list for %s:" % (s[i], m[marketid]))
        # data為字典格式，keys為字串格式，values為Rows格式(namedtuple)
        data = FetchItemList(urls[i], headers[i])
        sleep(mySet.RandTS()) # wait for sending next request
        # Save to .csv file
        SaveItemList(data, TW_List_File, myParams.Delimiter)
        Save_Counter += 1
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def CheckUserPool():
    """ 整理使用者提供的觀察池 """
    pool_new = []
    a = b = 0
    for code in User_Pool:
        if mySet.TWCodeCheck(code) == True: # Valid code
            try:
                Item_Dict[code] # Look up in the dict
                pool_new.append(code)
            except KeyError: # Invalid key
                b += 1
                pass
        else:
            a += 1
    # Print the removed "codes", then identify the cause:
    if len(pool_new) != len(User_Pool):
        items = [item for item in User_Pool if item not in pool_new]
        Logger.warning("No. of invalid code format: %s" % a)
        Logger.warning("No. of code not found in dict: %s" % b)
        Logger.warning("Show items: {}".format(' '.join(map(str, items))))
    return pool_new

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def TWSEPriceScraper(code, sd, ed):
    """
    爬取證交所資料(非還原權息股價)
    格式：日期 / 成交股數 / 成交金額 / 開盤價 / 最高價 / 最低價 / 收盤價 / 漲跌價差 / 成交筆數
    操作方式有兩種：
    (1) 透過Selenium點擊列印/HTML後，在列印網頁使用BeautifulSoup擷取資料
    (2) 直接提供列印網頁的url，並使用BeautifulSoup或Pandas擷取資料
    CSV下載的格式和編碼並不理想，故不考慮
    """
    
    # 由於證交所網頁一次只可取得一個月內的歷史成交資料，故需先建立迴圈以重複操作
    urllist = mySet.GenerateTWSEUrls(code, sd, ed)
    headers = mySet.TWSEPrice_Headers
    fpath = Raw_Path  + "\\TWSE-" + code + myParams.PriceDataExt
    full_data = None
    Logger.debug("Now fetch data for %s from %s to %s:" % (code, sd, ed))
    for url in urllist: # Fetch data from each url
        # pd.read_html將取得DataFrame的list，在此因一個url只有一張表，因此list只有一個元素
        # 9 columns, df.columns[x][0]對應表格標題，df.columns[x][1]對應欄位名稱
        # df.index並未被定義，即df.index[0] = 0, df.index[1] = 1, ...
        if headers == {} or headers is None:  
            Logger.warning("Request headers is not specified!")
            data = pd.read_html(requests.get(url).text)[0]
        else:
            data = pd.read_html(requests.get(url, headers = headers).text)[0]
        data.columns = mySet.TWSE_Cols # Change column names
        if full_data is None:
            full_data = data.copy()
        else:
            full_data = pd.concat([full_data, data], axis = 0)
        Logger.debug("From url: %s" % url)
        Logger.debug("Current # of data merged: %d." % len(full_data))
        sleep(mySet.RandTS()) # wait for sending next request
    Logger.debug("Finish fetching data from TWSE.")
    # 存檔前做點簡單的資料處理，剩下的丟給後處理
    # 透過函數映射，轉換日期格式成isoformat 'yyyy-mm-dd'
    full_data.loc[:, mySet.TWSE_Cols[0]] = \
        full_data.loc[:, mySet.TWSE_Cols[0]].apply(mySet.DateFmtConv)
    full_data.to_csv(fpath, index = False, header = True)
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def TPEXPriceScraper(code, sd, ed):
    """
    爬取櫃買中心資料(非還原權息股價)
    格式：日期 / 成交千股 / 成交千元 / 開盤 / 最高 / 最低 / 收盤 / 漲跌 / 筆數
    操作方式有兩種：
    (1) 透過Selenium點擊列印/匯出HTML後，在列印網頁使用BeautifulSoup擷取資料
    (2) 直接提供列印網頁的url，並使用BeautifulSoup或Pandas擷取資料
    CSV下載的格式並不理想，故不考慮 (但櫃買有提供utf-8編碼版本，可避免亂碼)
    TWSE與TPEX爬蟲方式相近，可以寫成單一函式，但為了避免日後兩邊網頁格式分岐，故仍寫成兩支函式
    """
    
    # 由於櫃買中心網頁一次只可取得一個月內的歷史成交資料，故需先建立迴圈以重複操作
    urllist = mySet.GenerateTPEXUrls(code, sd, ed)
    headers = mySet.TPEXPrice_Headers
    fpath = Raw_Path  + "\\TPEX-" + code + myParams.PriceDataExt
    full_data = None
    Logger.debug("Now fetch data for %s from %s to %s:" % (code, sd, ed))
    for url in urllist: # Fetch data from each url
        # pd.read_html將取得DataFrame的list，在此因一個url只有一張表，因此list只有一個元素
        # 9 columns, df.columns[x][0]對應表格標題，df.columns[x][1]對應欄位名稱
        # df.index並未被定義，即df.index[0] = 0, df.index[1] = 1, ...
        if headers == {} or headers is None:
            Logger.warning("Request headers is not specified!")
            data = pd.read_html(requests.get(url).text)[0]
        else:
            data = pd.read_html(requests.get(url, headers = headers).text)[0]
        data.columns = mySet.TPEX_Cols # Change column names
        if full_data is None:
            full_data = data.iloc[:-1, :].copy() # 不複製最後一列，那是沒用的註解
        else: # Merge data
            full_data = pd.concat([full_data, data.iloc[:-1, :]], axis = 0)
        Logger.debug("From url: %s" % url)
        Logger.debug("Current # of data merged: %d." % len(full_data))
        sleep(mySet.RandTS()) # wait for sending next request
    Logger.debug("Finish fetching data from TPEX.")
    # 存檔前做點簡單的資料處理，剩下的丟給後處理
    # 透過函數映射，轉換日期格式成isoformat 'yyyy-mm-dd'
    full_data.loc[:, mySet.TPEX_Cols[0]] = \
        full_data.loc[:, mySet.TPEX_Cols[0]].apply(mySet.DateFmtConv)
    full_data.to_csv(fpath, index = False, header = True)
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def YFinScraper(code_ex, sd = None, ed = None, IsDebugging = False):
    """ 
    爬取Yahoo Finance資料(非還原與還原權息股價與成交量)
    欄位：Date / Open / High / Low / Close* / Adj-close** / Volume
    Close price 包含對股本的調整
    Adj-close price 則包含對股本和股利的調整
    Volume的單位為股，一千股為一張
    
    時間週期為設定日期範圍，但須和MAX time period做比對，必要時調整範圍
    (和上市櫃時間 & 下市櫃時間比對，另外也可能有暫停交易的情形發生)
    欄位中會出現除權息資訊(dividend, split)；缺資料時會出現null
    商品項目整理於User_Pool，直接讀入
    台股資訊幣別預設為TWD，但國外商品須注意其對應的幣別
    
    2020.11.04 新增：
    發現指定日期範圍下有可能得不到今日的價格資訊，非常鳥！
    這時候換一個不指定日期範圍的 url，預設取得近一年的日K資訊
    此時才會得到今日資訊
    此函式若不提供起始與結束日期，則會改抓一年內歷史股價資訊
    """
    # Create an instance of the selenium webdriver 
    # The browser will be changed to PhantomJS in the future
    prefs = {'download.default_directory': Raw_Path}
    options = webdriver.ChromeOptions()
    options.add_experimental_option('prefs', prefs)
    if not IsDebugging: # 除錯完成後，瀏覽器採用無頭模式
        options.add_argument('--headless')
        # options.add_argument('--disable-gpu') # not used
    options.add_argument("--disable-extensions")
    driver = webdriver.Chrome(options = options, \
                              executable_path = myParams.Chrome_Path)
    # 隱性等待4-8秒，顯性最長8-16秒同時每0.5秒測試一次
    # 意即每次的等待秒數控制在4-8或8-16秒間
    t1 = mySet.RandTS()
    t2 = 2*t1
    driver.implicitly_wait(t1)
    # 省略透過selenium.webdriver按鈕操作，直接產生操作後的url
    # 如果這做法會被block IP，則需改採按鈕操作的方式進行
    url = mySet.GenerateYFUrl(code_ex)
    if sd != None and ed != None: # date range is specified!
        Logger.debug("Fetch data for %s from %s to %s:" % (code_ex, sd, ed))
        url = mySet.GenerateYFUrlTP(code_ex, sd, ed)
    else:
        Logger.debug("Date range is not specified!")
        Logger.debug("Fetch default date range for %s." % code_ex)
    driver.get(url)
    try:
        # All attribute::id = render-target-default located at any node
        locator = '//*[@id="render-target-default"]'
        wait = WebDriverWait(driver, t2, 0.5)
        element = \
            wait.until(EC.presence_of_element_located((By.XPATH, locator)))
        # 檢視網頁內容，得到Download對應的位置如下
        click_xpath = '//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[1]/div[2]/span[2]/a'
        dl_click = driver.find_element_by_xpath(click_xpath)
        target = driver.find_element_by_link_text(dl_click.text)
        driver.execute_script('arguments[0].scrollIntoView(true);', target)
        driver.execute_script('scrollBy(0, -500);') # Scroll up
        # driver.execute_script('scrollBy(0, -10000);') # Scroll to top
        sleep(mySet.RandTS())
        dl_click.click() # Download .csv historical data
        sleep(mySet.RandTS())
        Logger.debug("Finish fetching data from Yahoo Finance.")
    except TimeoutException:
        Logger.error("Page is not ready within timeout limit!")
    finally:
        driver.quit() # close the browser
    # 檢查檔案是否存在
    fpath = Raw_Path  + '\\' + code_ex + myParams.PriceDataExt
    if not os.path.isfile(fpath):
        Logger.warning("%s is not a file!" % fpath)
    else:
        fs = os.path.getsize(fpath) # file size in bytes
        if fs == 0:
            Logger.warning("%s is empty!" % fpath)      
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def GetPriceData(datasrc, sd, ed, isyfdefault = False):
    """
    提供清單，爬取清單內各商品價量資訊
    參數：
    資料來源、起始日、結束日、是否用預設日期範圍(Yahoo Finance only)
    """
    Logger.info("Data source for scraping: %s" % datasrc)
    # Argument validation:
    if datasrc == mySet.PriceDataSrc[0]:
        for code in User_Pool:
            market = Item_Dict[code].Market
            if market == "上市":
                TWSEPriceScraper(code, sd, ed)
            elif market == "上櫃":
                TPEXPriceScraper(code, sd, ed)
            else:
                Logger.warning("code %s (market %s) is not scraped!" \
                               % (code, market))
    elif datasrc == mySet.PriceDataSrc[1]:
        for code in User_Pool:
            market = Item_Dict[code].Market
            if market == "上市":
                if isyfdefault == True: 
                    YFinScraper(str(code) + ".TW",)
                else:
                    YFinScraper(str(code) + ".TW", sd, ed,)
            elif market == "上櫃":
                if isyfdefault == True:
                    YFinScraper(str(code) + ".TWO",)
                else:
                    YFinScraper(str(code) + ".TWO", sd, ed,)
            else:
                Logger.warning("code %s (market %s) is not scraped!" \
                               % (code, market))
    else:
        Logger.warning("Unknown data source! Exit directly!")
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def Main(infname, lastdate, numdays, pdatasrc, isyfdefault = False):
    """
    主函式: 須提供股票清單與資料擷取日期範圍
    時間採用isoformat：YYYY-MM-DD
    """
    # Set path and folders:
    if not os.path.isdir(Raw_Path):
        os.mkdir(Raw_Path)
    
    # Update TWSE and TPEX lists:
    global Save_Counter
    Save_Counter = 0 # 重設計數器，第一次存清單時將覆蓋掉舊的
    UpdateItemList(1) # 上市商品
    UpdateItemList(2) # 上櫃商品
    
    # Setup item dict
    global Item_Dict
    Item_Dict = {}
    LoadItemList(TW_List_File, myParams.Delimiter)
    
    # Read user assigned list (pool)
    global User_Pool
    User_Pool = []
    with open(infname, 'r', encoding = "utf-8") as f:
        User_Pool = f.read().splitlines()
    User_Pool = CheckUserPool() # Remove invalid codes
    
    # Start scraping!
    ed = dt.date.isoformat(lastdate) # 或用str(lastdate)
    sd = dt.date.isoformat(lastdate - dt.timedelta(days = numdays))
    Logger.info("Date range: %s - %s" % (sd, ed)) # YYYY-MM-DD
    # 用兩個資料來源之一擷取資料
    # 未來可改成：短天期爬證交所，長天期爬Yahoo財經，作為資料質與量的取捨
    if pdatasrc == mySet.PriceDataSrc[0]:
        GetPriceData(pdatasrc, sd, ed,) # TWSE & TPEX
        Logger.info("Scrape data from TWSE and TPEX.")
    elif pdatasrc == mySet.PriceDataSrc[1]:
        GetPriceData(pdatasrc, sd, ed, isyfdefault) # Yahoo finance
        Logger.info("Scrape data from Yahoo Finance.")
    else:
        Logger.warning("Undefined price data source %s!" % pdatasrc)
        Logger.warning("Skip web scraping for historical data ...")
    # Post-processing: see myPostProcessing.py
    return None

# Function testing 
if __name__ == "__main__":
    t_start = dt.datetime.now()
    print("***** Module myScraper test: *****")
    print("Show ref. path: ", RefPath)
    demo_file = RefPath + '\\' + myParams.Test_Infile
    if not os.path.isfile(demo_file):
        print("File %s does not exist! Create one!" % demo_file)
        with open(demo_file, "w") as f:
            f.write("2330\n4743") # give two items for scraping
            # f.write("2330\n4743\n0050\n00677U")
    dayspan = 60
    enddate = dt.date.today()
    # dayspan = 150
    # enddate = dt.date.fromisoformat('2020-10-20')
    startdate = enddate - dt.timedelta(days = dayspan)
    print("Infile: %s\nDate range: From %s to %s\n# of days: %d." %\
          (demo_file, str(startdate), str(enddate), dayspan))
    Main(demo_file, enddate, dayspan, mySet.PriceDataSrc[0],)
    Main(demo_file, enddate, dayspan, mySet.PriceDataSrc[1],)
    # Main(demo_file, enddate, dayspan, mySet.PriceDataSrc[1], True)
    t_end = dt.datetime.now()
    print("Total ellapsed time is: ", t_end - t_start)
# Done!
