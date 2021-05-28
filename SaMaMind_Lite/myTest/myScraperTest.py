# -*- coding: utf-8 -*-

import os
import datetime as dt
from time import sleep
import pandas as pd
import unittest
from myStockScraper import myScraper
from myStockScraper.myScraper import mySet as mySet
from myUtilities import myLog

# --- --- #

# 測試項目篩選設定
defaultReason = "Assigned by tester"
skip_TestLogger = [False, defaultReason]
skip_TestListFetchers = [False, defaultReason]
skip_TestListOperation = [False, defaultReason]
skip_TestScrapers = [False, defaultReason]
skip_TestMain = [False, defaultReason]

# 存取檔案的預設父路徑
defaultPath = os.path.dirname(os.getcwd())

# 設定單元測試日誌
# LogName = None # root logger
LogName = "myScraper_Unittest"
Logger = myLog.Log(myLog.LogLevel.DEBUG, LogName)

# 建立兩組User_Pool，其中一個內容格式是合法的，另一個是不合法的
InStr = ["2330\n1402\n4743\n0050\n00677U\n006201", 
         "aaaa\n123\nq1w2e3\n123456789\n0123"]
for i in range(0, len(InStr)):
    testInF = defaultPath + "\\myScraperTest-" + str(i+1) + ".in"
    if not os.path.isfile(testInF):
        with open(testInF, "w") as f:
            f.write(InStr[i])

# 測試用清單
test_dict = {
    "0050": mySet.Rows("0050", "元大台灣50", "上市", "ETF"),
    "2330": mySet.Rows("2330", "台積電", "上市", "股票"),
    "1402": mySet.Rows("1402", "遠東新", "上市", "股票"),
    "4743": mySet.Rows("4743", "合一", "上櫃", "股票"),
    "00677U": mySet.Rows("00677U", "富邦VIX", "上市", "ETF"),
    "006201": mySet.Rows("006201", "元大富櫃50", "上櫃", "ETF")}

# --- --- #

@unittest.skipIf(skip_TestLogger[0], skip_TestLogger[1])
class TestLogger(unittest.TestCase):
    """ 進行myScraper模組中日誌操作測試，日誌功能來源為myLog模組 """
    def setUp(self):
        Logger.debug("Setup: TestLogger.")
    def tearDown(self):
        Logger.debug("Tearing down the TestLogger test.")
    def test_logName(self):
        Logger.debug(" *** test_logName *** ")
        fFame = "myStockScraper.myScraper"
        self.assertEqual(fFame, myScraper.LogName)
    def test_logger(self):
        Logger.debug(" *** test_logger *** ")
        myScraper.Logger.info("Call logger via unit test!")
        fPath = defaultPath + "\\log\\myStockScraper.myScraper.log"
        isLogExist = os.path.isfile(fPath)
        self.assertTrue(isLogExist)
    def test_funLogger(self):
        Logger.debug(" *** test_funLogger *** ")
        deLim = ','
        inF = defaultPath + "\\myScraperTest-Save.csv"
        myScraper.SaveItemList(test_dict, inF, deLim)
        fPath = defaultPath + "\\log\\myStockScraper.myScraper.SaveItemList.log"
        isFunLogExist = os.path.isfile(fPath)
        self.assertTrue(isFunLogExist)
    def test_errLogger(self):
        Logger.debug(" *** test_errLogger *** ")
        url = '' # To raise exception
        headers = {}
        myScraper.FetchItemList(url, headers)
        fPath = defaultPath + "\\log\\Err.log"
        isErrLogExist = os.path.isfile(fPath)
        self.assertTrue(isErrLogExist)   

@unittest.skipIf(skip_TestListFetchers[0], skip_TestListFetchers[1])
class TestListFetchers(unittest.TestCase):
    """ Fetcher測試：確認可以與各站台成功請求資料 """
    def setUp(self):
        Logger.debug("Setup: TestListFetchers.")
        self.url = ''
        self.headers = None
        self.msg = ''
    def tearDown(self):
        Logger.debug("Tearing down the TestListFetchers test.")
    def test_Fetchers(self):
        Logger.debug(" *** test_Fetchers *** ")
        url = [mySet.TWSE_Stock_Url, 
               mySet.TPEX_Stock_Url,
               mySet.TWSE_ETF_Url,
               mySet.TPEX_ETF_Url]
        headers = [mySet.TWSE_Stock_Headers,
                   mySet.TPEX_Stock_Headers,
                   mySet.TWSE_ETF_Headers,
                   mySet.TPEX_ETF_Headers]
        msg = ["Market = TWSE, Sec. type = stock", 
               "Market = TPEX, Sec. type = stock", 
               "Market = TWSE, Sec. type = ETF", 
               "Market = TPEX, Sec. type = ETF"]
        for u, h, m in zip(url, headers, msg):
            self.url = u
            self.headers = h
            self.msg = m
            with self.subTest():
                Logger.debug("Subtest for: %s." % self.msg)
                data = myScraper.FetchItemList(self.url, self.headers)
                Logger.debug("# rows of data: %d." % len(data))
                counts = 0
                for k in data.keys():
                    if counts < 3:
                        a = data[k].TW_Code
                        b = data[k].Name
                        c = data[k].Market
                        d = data[k].Sec_Type
                        Logger.debug("Search key: %s" % k)
                        Logger.debug("Find: %s-%s-%s-%s." % (a, b, c, d))
                    counts += 1
                sleep(3)
                self.assertIsNotNone(data)
                self.assertIsInstance(data, dict) # not a str
                self.assertTrue(len(data) > 1)

@unittest.skipIf(skip_TestListOperation[0], skip_TestListOperation[1])
class TestListOperation(unittest.TestCase):
    """ Item list操作測試 """
    def setUp(self):
        Logger.debug("Setup: TestListOperation.")
    def tearDown(self):
        Logger.debug("Tearing down the TestListOperation test.")
    def test_Update_Items(self):
        Logger.debug(" *** test_Update_Items *** ")
        for i in range(0, 2):
            with self.subTest(i = i):
                myScraper.UpdateItemList(i+1)
                myScraper.Save_Counter = 0 # Clean-up
                self.assertTrue(os.path.isfile(myScraper.TW_List_File))
                self.assertTrue(os.path.getsize(myScraper.TW_List_File) > 0)
    def test_Load_Items(self):
        Logger.debug(" *** test_Load_Items *** ")
        for i in range(0, 2):
            myScraper.UpdateItemList(i+1)
        myScraper.Save_Counter = 0 # Clean-up
        myScraper.LoadItemList(myScraper.TW_List_File, ',')
        testItemDict = myScraper.Item_Dict.copy()
        myScraper.Item_Dict = {} # Clean-up
        self.assertTrue(len(testItemDict) > 1)
        self.assertIsInstance(testItemDict, dict)
        self.assertEqual(testItemDict['2330'].TW_Code, '2330')
        self.assertEqual(testItemDict['0050'].Sec_Type, 'ETF')
        self.assertEqual(testItemDict['4743'].Sec_Type, '股票')
        self.assertEqual(testItemDict['2330'].Market, '上市')
        # self.assertEqual(testItemDict['0050'].Market, '上市')
        # self.assertEqual(testItemDict['4743'].Market, '上櫃')
    def test_User_Pool(self):
        Logger.debug(" *** test_User_Pool *** ")
        for key in test_dict.keys():
            myScraper.User_Pool.append(key)
        myScraper.Item_Dict = test_dict.copy()
        testPoolNew = myScraper.CheckUserPool()
        testUserPool = myScraper.User_Pool.copy()
        myScraper.User_Pool = [] # Clean-up
        myScraper.Item_Dict = {} # Clean-up
        self.assertTrue(len(testPoolNew) > 0)
        self.assertEqual(testUserPool, testPoolNew)
        self.assertIsInstance(testPoolNew, list)
    def test_Load_Pool(self):
        """ 與test_User_Pool功能接近，只差在pool的取得方式不同 """
        Logger.debug(" *** test_Load_Pool *** ")
        for i in range(0, len(InStr)):
            with self.subTest(i = i):
                inf = defaultPath + "\\myScraperTest-" + str(i+1) + ".in"
                with open(inf, 'r', encoding = "utf-8") as f:
                    myScraper.User_Pool = f.read().splitlines()
                myScraper.Item_Dict = test_dict.copy()
                testLoadPool = myScraper.User_Pool.copy()
                testPoolNew = myScraper.CheckUserPool()
                myScraper.User_Pool = [] # Clean-up
                myScraper.Item_Dict = {} # Clean-up
                if i == 0:
                    Logger.debug("All elements in pool should be retained!")
                    self.assertTrue(len(testPoolNew) > 0)
                    self.assertEqual(testLoadPool, testPoolNew)
                    self.assertIsInstance(testPoolNew, list)
                elif i == 1:
                    Logger.debug("All elements in pool should be removed!")                    
                    self.assertTrue(len(testLoadPool) > 0)
                    self.assertEqual(len(testPoolNew), 0)

@unittest.skipIf(skip_TestScrapers[0], skip_TestScrapers[1])
class TestScrapers(unittest.TestCase):
    """ 測試所有的網路爬蟲，用於擷取歷史股價和成交量資訊 """
    def setUp(self):
        self.sd = "2020-04-29" # 交易日
        self.ed = "2020-10-29" # 交易日
        Logger.debug("Setup: TestScrapers.")
        Logger.info("Date range: from %s to %s." % (self.sd, self.ed))
    def tearDown(self):
        Logger.debug("Tearing down the TestScrapers test.")
    def test_CheckPath(self):
        Logger.debug(" *** test_CheckPath *** ")
        dPath = defaultPath + "\\data scraped"
        self.assertEqual(myScraper.Raw_Path, dPath)
    def test_TWSEPriceScraper(self):
        Logger.debug(" *** test_TWSEPriceScraper *** ")
        code = "5234"
        fPath = myScraper.Raw_Path + "\\TWSE-" + code + '.csv'
        myScraper.TWSEPriceScraper(code, self.sd, self.ed)
        self.assertTrue(os.path.isfile(fPath))
        self.assertTrue(os.path.getsize(fPath) > 0)
    def test_TPEXPriceScraper(self):
        Logger.debug(" *** test_TPEXPriceScraper *** ")
        code = "4743"
        fPath = myScraper.Raw_Path + "\\TPEX-" + code + '.csv'
        myScraper.TPEXPriceScraper(code, self.sd, self.ed)
        self.assertTrue(os.path.isfile(fPath))
        self.assertTrue(os.path.getsize(fPath) > 0)
    def test_YFPriceScraper(self):
        Logger.debug(" *** test_YFPriceScraper *** ")
        code_ex = ["5234.TW", "4743.TWO"]
        isdebug = False
        for i in range(0, len(code_ex)):
            with self.subTest(i = i):
                fPath = myScraper.Raw_Path + '\\' + code_ex[i] + '.csv'
                myScraper.YFinScraper(code_ex[i], self.sd, self.ed, isdebug)
                self.assertTrue(os.path.isfile(fPath))
                self.assertTrue(os.path.getsize(fPath) > 0)
    def test_YFPriceScraper_MaxTP(self):
        Logger.debug(" *** test_YFPriceScraper_MaxTP *** ")
        code_ex = "00677U.TW"
        sd = "2012-01-01" # 早於發行日
        ed = str(dt.date.today()) # 今日
        isdebug = True
        fPath = myScraper.Raw_Path + "\\00677U.TW.csv"
        myScraper.YFinScraper(code_ex, sd, ed, isdebug)
        self.assertTrue(os.path.isfile(fPath))
        self.assertTrue(os.path.getsize(fPath) > 0)
    def test_YFPriceScraper_NoTP(self):
        Logger.debug(" *** test_YFPriceScraper_NoTP *** ")
        code_ex = "2330.TW"
        fPath = myScraper.Raw_Path + "\\2330.TW.csv"
        myScraper.YFinScraper(code_ex, )
        df = pd.read_csv(fPath, sep = ',', header = 0, index_col = None)
        self.assertTrue(os.path.isfile(fPath))
        self.assertTrue(os.path.getsize(fPath) > 0)
        # 若今日非交易日則會出錯
        self.assertEqual(df.iloc[-1, 0], str(dt.date.today()))

@unittest.skipIf(skip_TestMain[0], skip_TestMain[1])
class TestMain(unittest.TestCase):
    """
    測試模組的主函式，功能除了GetPriceData()部分外與上述幾乎重複
    可與Function testing(執行myScraper模組的主函式)做比較
    """
    def test_Main_No_Asserts(self):
        demo_file = defaultPath + "\\unittest.in"
        content = []
        for key in test_dict.keys():
            content.append(key)
        if not os.path.isfile(demo_file):
            with open(demo_file, "w") as f:
                for i in range(0, len(content)):
                    f.write(content[i] + '\n')
        dayspan = 7
        enddate = dt.date.today()
        myScraper.Main(demo_file, enddate, dayspan, mySet.PriceDataSrc[0], )
        myScraper.Main(demo_file, enddate, dayspan, \
                       mySet.PriceDataSrc[1], True) # 預設日期範圍
        # No asserts

# --- --- #
if __name__ == "__main__":
    unittest.main()
# Done!
