# -*- coding: utf-8 -*-

import os
from shutil import rmtree
from time import sleep
import datetime as dt
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.ttk as ttk # Tk themed widget set (tk ver. > 8.5)
import threading
import webbrowser
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import re

from myStockScraper import myScraper
from myStockScraper import myScraperSettings as mySet
from myStockScraper import myPostProcessing as myPostP
from myUtilities import myParams
from myUtilities import myTechIndices as myTI
from myMind import myWorld, myEnv, myAgent, myAna
from myMind import myDNNVisualizer

"""
myApp.py 的功能為 GUI 主視窗與功能對應
"""

# Global variables and constants:
RefPath = '' # 參考路徑
if __name__ == "__main__": # executed
    RefPath = os.path.dirname(os.getcwd())
else: # imported
    modulepath = os.path.dirname(__file__)
    RefPath = os.path.dirname(modulepath)

UserInfPath = RefPath + '\\' + myParams.User_Inf_Folder
UserInfPath_Scraper = UserInfPath + "\\user pool"
UserInfPath_RLParams = UserInfPath + "\\ai param set"
if not os.path.isdir(UserInfPath):
    os.mkdir(UserInfPath)
if not os.path.isdir(UserInfPath_Scraper):
    os.mkdir(UserInfPath_Scraper)
if not os.path.isdir(UserInfPath_RLParams):
    os.mkdir(UserInfPath_RLParams)

ReadmePath = RefPath + "\\readme.txt"
LogFolder = RefPath + "\\log"

# 目前不開放外部更改以下兩參數：
TD = myParams.TradePeriods[1]
# KWList = list(myParams.DefinedTechInd)
KWList = ["MACD", "SMA", "CVOL"]

IcoJpgPath = RefPath + "\\images\\sama_ico.jpg" # Icon圖檔
IconPhoto = None

Default_Width = 1200
Default_Height = 800
Default_OffsetX = 100
Default_OffsetY = 100
Default_ResizeF = 0.075

GetListContent = ''

# --- 輔助函式 --- #

def _GenIconPhoto(isreturn = False):
    """ 產生視窗圖示 """
    global IconPhoto
    w_ar = float(Default_Width)/float(Default_Height) # 視窗長寬比
    try: # 開啟圖檔並轉檔
        _img = Image.open(IcoJpgPath) # return a image object
        w0, h0 = _img.size
        i_ar = float(w0)/float(h0) # 圖片長寬比
        w_i_ratio = w_ar/i_ar
        # 假若GUI視窗較寬則圖片會被拉長，故須修正保持圖片長寬比例
        w = int(Default_Width * Default_ResizeF / w_i_ratio)
        h = int(Default_Height * Default_ResizeF)
        _img = _img.resize((w, h), Image.ANTIALIAS) # 縮放圖示
        _photo = ImageTk.PhotoImage(_img)
        # _photo = ImageTk.PhotoImage(Image.fromarray(_img))
    except Exception as e:
        IconPhoto = None
        tk.messagebox.showerror("Icon圖檔轉換錯誤", e)
    else:
        IconPhoto = _photo
    if isreturn:
        return IconPhoto
    else:
        return None

def SetIcon(master):
    """ 變更視窗圖示 """
    _GenIconPhoto(False)
    sleep(0.5)
    if not IconPhoto is None:
        try:
            master.tk.call("wm", "iconphoto", \
                           master._w, IconPhoto)
        except Exception as e:
            tk.messagebox.showerror("Icon設定錯誤", e)

def OpenLink(url):
    """ 使用預設網路瀏覽器開啟網頁 """
    try:
        webbrowser.open_new(url)
    except Exception:
        msg = "非預期錯誤！\n發生於開啟連結：\n" + url
        tk.messagebox.showerror("不明錯誤", msg)

def OpenFile(fpath):
    """ (不透過GUI對話視窗)直接用預設程式打開指定檔案 """
    try:
        os.startfile(fpath)
    except FileNotFoundError:
        msg = "找不到以下檔案：\n" + fpath
        tk.messagebox.showerror("檔案不存在", msg)
    except OSError:
        msg = "無法開啟：\n" + fpath
        tk.messagebox.showerror("開啟檔案失敗", msg)
    except Exception:
        msg = "非預期錯誤！\n發生於開啟檔案：\n" + fpath
        tk.messagebox.showerror("不明錯誤", msg)

def ExitApp(master):
    """ 關閉主視窗 """
    msgbox = tk.messagebox.askquestion("離開程式", "請確認是否離開？")
    if msgbox == "yes":
        master.destroy()
    # else, return to the main window

def TextReset(text_widget, text, tags):
    """ 清除文字編輯器內容，換成預設值 """
    try:
        text_widget.delete(0.0, tk.END)
        if not tags is None:
            text_widget.insert(0.0, text, tags)
        else:
            text_widget.insert(0.0, text)
    except Exception as e:
        # 最容易出錯的原因可能是 tags 未定義
        tk.messagebox.showerror("載入預設值錯誤", e)

def TSVReset(tkstrvar, resetval):
    """ 將tk.StringVar透過set方法還原成預設值 """
    tkstrvar.set(resetval)

# --- GUI相關類別 --- #

class TkTab(ttk.Frame):
    """ 用於新增至分頁籤的框架類別 """
    _type_id_min = 0
    _type_id_max = 5
    def __init__(self, master, type_id, *args, **kwargs):
        """
        框架初始化， 其中各引數為：
        (元件 = Widget, 框架 = Frame, 頁面 = Tab)
            master (tk.Toplevel) - 為 Frame widget 的父類別(主視窗)
            type_id (int) - 指定建立框架的方法(在此為ttk.notebook之頁面)
        """
        super().__init__(master, *args, **kwargs)
        self.type_id = type_id
        self._validate_typeid()
        if self.type_id == 0:
            self._create_type0(master)
        elif self.type_id == 1:
            self._create_type1(master)
        elif self.type_id == 2:
            self._create_type2(master)
        elif self.type_id == 3:
            self._create_type3(master)
        elif self.type_id == 4:
            self._create_type4(master)
        # else, do nothing
    
    def _validate_typeid(self):
        """ 輔助函式，用於限制type_id上下界 """
        if self.type_id > self._type_id_max:
            self.type_id = self._type_id_max
        elif self.type_id < self._type_id_min:
            self.type_id = self._type_id_min
    
    def _create_type0(self, master):
        """ 建立第一個頁面內容；引數 master 用於主視窗操作 """
        ### 程式標題
        self.ttl = tk.Text(self) # 多行文字標籤
        self.ttl.place(relx = 0.1, rely = 0.08, \
                       relwidth = 0.8, relheight = 0.08)
        self.ttl.tag_add("tagv", 0.0)
        self.ttl.tag_config("tagv", background = "papaya whip", \
                            foreground = "sienna", \
                            justify = "center", \
                            font = ("Microsoft JhengHei UI", 24, "bold"), \
                            relief = "ridge", wrap = "word")
        ttl = "～歡迎使用SaMaMind(Lite)～"
        self.ttl.insert(tk.END, ttl, ("tagv"))
        self.ttl["state"] = "disabled"
        
        ### 版本資訊
        # 避免scrollbar與text editor重疊遮住內容
        # 將兩者放在共同的框架上並排，而非把scrollbar放在text editor內
        self.vframe = ttk.Frame(self, borderwidth = 1, relief = "sunken")
        self.vframe.place(relx = 0.1, rely = 0.2, \
                          relwidth = 0.8, relheight = 0.6)
        self.vinfo = tk.Text(self.vframe, borderwidth = 0) # 多行文字標籤
        self.vinfo.pack(side = "left", fill = "both", expand = 1)
        self.vscroll = ttk.Scrollbar(self.vframe, cursor = "arrow")
        self.vscroll.pack(side = "right", fill = 'y')
        self.vinfo["yscrollcommand"] = self.vscroll.set # 連結捲動條與文字標籤
        # 當Scrollbar被捲動時，呼叫Text元件之視窗垂直捲動方法
        self.vscroll["command"] = self.vinfo.yview
        
        # 設定標籤
        self.vinfo.tag_add("tage", 0.0) # 錯誤訊息用
        self.vinfo.tag_config("tage", foreground = "pale violet red", \
                              underline = True, \
                              font = ("helvetica", 12, "italic", "bold"))
        self.vinfo.tag_add("tagi", 0.0) # readme內容
        self.vinfo.tag_config("tagi", foreground = "dark slate blue", \
                              background = "snow", \
                              font = ("Microsoft JhengHei UI", 12, "bold"))
        try:
            f = open(ReadmePath, 'r', encoding = "utf-8")
        except OSError: # include time-out, file-not-found, etc.
            self.vinfo.insert(tk.END, "Attempt to open: \n" + ReadmePath, \
                              ("tage"))
            self.vinfo.insert(tk.END, "\nFile cannot be opened!\n", \
                              ("tage"))
        except Exception as e:
            self.vinfo.insert(tk.END, "Attempt to open: \n" + ReadmePath, \
                              ("tage"))
            self.vinfo.insert(tk.END, "\nUnexpected problem occurred:\n", \
                              ("tage"))
            self.vinfo.insert(tk.END, e, ("tage"))
        else:
            self.vinfo.insert(tk.END, f.read(), ("tagi"))
        
        ### 開啟日誌檔
        self.btn_log = ttk.Button(self, text = "開啟日誌檔")
        self.btn_log.bind("<Button-1>", lambda e: OpenFile(LogFolder))
        self.btn_log.place(relx = 0.1, rely = 0.82, \
                           relwidth = 0.3, relheight = 0.06)
        
        ### 拜訪作者
        self.author = ttk.Label(self, text = "  作者：CafeYuzuHuang", \
                                image = IconPhoto, compound = "left", \
                                relief = "flat", background = "alice blue", \
                                foreground = "royal blue")
        self.author.place(relx = 0.6, rely = 0.82, \
                          relwidth = 0.3, relheight = 0.1)
        self.ghlink = ttk.Label(self, text = "CafeYuzuHuang @ github", \
                                foreground = "blue", background = "alice blue", \
                                cursor = "hand2", relief = "flat", \
                                font = ("Microsoft JhengHei UI", 9, "underline"))
        self.ghlink.place(relx = 0.7, rely = 0.92, \
                          relwidth = 0.2, relheight = 0.04)
        # 繫結連結網址
        self.ghlink.bind("<Button-1>", lambda e: \
                         OpenLink("https://github.com/CafeYuzuHuang"))
        
        ### 離開程式
        self.btn_quit = ttk.Button(self, text = "離開程式")
        self.btn_quit.bind("<Button-1>", lambda e: ExitApp(master))
        self.btn_quit.place(relx = 0.1, rely = 0.9, \
                            relwidth = 0.3, relheight = 0.06)
    
    def _create_type1(self, master):
        """ 建立第二個頁面內容，這裡處理爬蟲和後處理相關功能 """
        # 參數初始化
        self.infplist = [] # .in file 路徑清單
        self.infnlist = [] # .in file 檔名清單
        self.infploaded = '' # 當前載入的檔案路徑
        self.infcontent = '' # 當前載入的檔案內容
        self.save_path = UserInfPath_Scraper
        self.open_path = UserInfPath_Scraper
        self.default_infname = "demo_list.in"
        
        self.status_ttl = ttk.Label(self, text = " 狀態 ")
        self.status_ttl.place(relx = 0.03, rely = 0.9, \
                              relwidth = 0.06, relheight = 0.06)
        self.s = tk.StringVar()
        self.s.set("就緒")
        self.status = ttk.Label(self, textvariable = self.s, \
                                relief = "sunken", background = "ivory", \
                                foreground = "dark slate blue", \
                                padding = 5, \
                                font = ("Microsoft JhengHei UI", 12, \
                                        "bold", "italic"))
        self.status.place(relx = 0.09, rely = 0.9, \
                          relwidth = 0.85, relheight = 0.06)
        # 當狀態列Label元件變成可視時觸發事件，將內容還原成預設值
        self.status.bind("<Visibility>", \
                         lambda e: TSVReset(self.s, "就緒"))
        
        ### 建立自選清單
        self.getlist_ttl = ttk.Label(self, text = "建立自選清單")
        self.getlist_ttl.place(relx = 0.06, rely = 0.08, \
                               relwidth = 0.42, relheight = 0.06)
        # undo = True 表示允許Ctrl-Z/Shift-Ctrl-Z
        self.getlist = tk.Text(self, undo = True, \
                               foreground = "dark slate blue", \
                               background = "snow", \
                               font = ("Microsoft JhengHei UI", 12, "bold"))
        self.getlist.place(relx = 0.06, rely = 0.16, \
                           relwidth = 0.42, relheight = 0.44)
        self.vscroll = ttk.Scrollbar(self.getlist, cursor = "arrow")
        self.vscroll.pack(side = "right", fill = 'y')
        self.getlist["yscrollcommand"] = self.vscroll.set
        self.vscroll["command"] = self.getlist.yview
        
        # 設定標籤
        self.getlist.tag_add("tage", 0.0) # 錯誤訊息用
        self.getlist.tag_config("tage", foreground = "pale violet red", \
                                underline = True, \
                                font = ("helvetica", 12, "italic", "bold"))
        self.getlist.tag_add("tagi", 0.0) # readme內容
        self.getlist.tag_config("tagi", foreground = "dark slate blue", \
                                background = "snow", \
                                font = ("Microsoft JhengHei UI", 12, "bold"))
        
        # 當Text元件變成可視時觸發事件，將內容還原成預設值
        s1 = "請依序輸入證券代號\n例：\n1234\n5678\n..."
        self.getlist.bind("<Visibility>", \
                          lambda e: TextReset(self.getlist, s1, "tagi"))
        
        self.btn_save_in = ttk.Button(self, text = "儲存自選清單", \
                                      command = self.saveinf)
        self.btn_save_in.place(relx = 0.06, rely = 0.62, \
                               relwidth = 0.2, relheight = 0.06)
        
        ### 載入自選清單
        self.btn_open_in = ttk.Button(self, text = "載入自選清單", \
                                      command = self.openinf)
        self.btn_open_in.place(relx = 0.28, rely = 0.62, \
                               relwidth = 0.2, relheight = 0.06)
        
        ### 資料更新設定
        # myScraper需要的參數包含：
        # infile路徑、結束日、天期、站台來源
        # 預設 = 抓取一年份資料
        self.a = tk.StringVar()
        self.a.set(mySet.PriceDataSrc[0]) # 預設：證交所
        self.b = tk.StringVar()
        self.b.set(str(dt.date.today())) # or dt.date.isoformat(dt.date.today())
        self.c = tk.IntVar()
        self.c.set(180)
        self.d = tk.BooleanVar()
        self.d.set(False)
        
        self.rb_ttl = ttk.Label(self, text = "證券資料來源")
        self.rb_ttl.place(relx = 0.52, rely = 0.08, \
                          relwidth = 0.2, relheight = 0.14)
        src = mySet.PriceDataSrc
        self.rba = ttk.Radiobutton(self, text = "證交所&櫃買中心", \
                                   variable = self.a, value = src[0])
        self.rba.place(relx = 0.74, rely = 0.08, \
                       relwidth = 0.2, relheight = 0.06)
        self.rbb = ttk.Radiobutton(self, text = "Yahoo Finance", \
                                   variable = self.a, value = src[1])
        self.rbb.place(relx = 0.74, rely = 0.16, \
                       relwidth = 0.2, relheight = 0.06)
        
        self.ymd_ttl = ttk.Label(self, text = "最後交易日\n(yyyy-mm-dd)")
        self.ymd_ttl.place(relx = 0.52, rely = 0.24, \
                           relwidth = 0.2, relheight = 0.10)
        self.ymd = ttk.Entry(self, textvariable = self.b)
        self.ymd.place(relx = 0.74, rely = 0.26, \
                       relwidth = 0.2, relheight = 0.06)
        
        self.days_ttl = ttk.Label(self, text = "橫跨天數")
        self.days_ttl.place(relx = 0.52, rely = 0.42, \
                            relwidth = 0.3, relheight = 0.06)
        self.days = ttk.Entry(self, textvariable = self.c)
        self.days.place(relx = 0.74, rely = 0.42, \
                        relwidth = 0.2, relheight = 0.06)
        
        chktext = "使用預設日期：本日往前算一年"
        self.defday = ttk.Checkbutton(self, text = chktext, \
                                      variable = self.d, \
                                      onvalue = True, offvalue = False, \
                                      command = self.disable_ent_ymd)
        self.defday.place(relx = 0.6, rely = 0.5, \
                          relwidth = 0.3, relheight = 0.06)
        
        ### 資料更新
        self.fun_handler = self.scraper # th_run 目標函式
        self.btn_scraper_go = ttk.Button(self, text = "開始下載\n更新資料", \
                                         command = self.th_run)
        self.btn_scraper_go.place(relx = 0.74, rely = 0.74, \
                                  relwidth = 0.2, relheight = 0.1)
        self.btn_scraper_go["state"] = "disabled"
        # self.s.set("就緒") # 結束
    
    def _create_type2(self, master):
        """
        建立第三個頁面內容，這裡處理模型訓練相關功能
        前半段的寫法和上一個函式幾乎一樣
        """
        # 參數初始化
        self.infplist = [] # .in file 路徑清單
        self.infnlist = [] # .in file 檔名清單
        self.infploaded = '' # 當前載入的檔案路徑
        self.infcontent = '' # 當前載入的檔案內容
        self.save_path = UserInfPath_RLParams
        self.open_path = UserInfPath_RLParams
        self.default_infname = "default_pset.in"
        self.default_psetname = "default_test" # 日後可以改成別的
        self.inpset_dict = {} # 載入的參數組
        self.dfpset_dict = {}
        self.model_name = "demo_test" # RL agent 模型名稱 (副檔名為.h5)
        
        self.status_ttl = ttk.Label(self, text = " 狀態 ")
        self.status_ttl.place(relx = 0.03, rely = 0.9, \
                              relwidth = 0.06, relheight = 0.06)
        self.s = tk.StringVar()
        self.s.set("就緒")
        self.status = ttk.Label(self, textvariable = self.s, \
                                relief = "sunken", background = "ivory", \
                                foreground = "dark slate blue", \
                                padding = 5, \
                                font = ("Microsoft JhengHei UI", 12, \
                                        "bold", "italic"))
        self.status.place(relx = 0.09, rely = 0.9, \
                          relwidth = 0.85, relheight = 0.06)
        self.status.bind("<Visibility>", \
                         lambda e: TSVReset(self.s, "就緒"))
        
        ### 建立訓練參數組
        self.getlist_ttl = ttk.Label(self, text = "建立訓練參數組")
        self.getlist_ttl.place(relx = 0.06, rely = 0.08, \
                               relwidth = 0.42, relheight = 0.06)
        self.getlist = tk.Text(self, undo = True, \
                               foreground = "dark slate blue", \
                               background = "snow", \
                               font = ("Microsoft JhengHei UI", 12, "bold"))
        self.getlist.place(relx = 0.06, rely = 0.16, \
                           relwidth = 0.42, relheight = 0.48)
        self.vscroll = ttk.Scrollbar(self.getlist, cursor = "arrow")
        self.vscroll.pack(side = "right", fill = 'y')
        self.getlist["yscrollcommand"] = self.vscroll.set
        self.vscroll["command"] = self.getlist.yview
        
        # 設定標籤
        self.getlist.tag_add("tage", 0.0) # 錯誤訊息用
        self.getlist.tag_config("tage", foreground = "pale violet red", \
                                underline = True, \
                                font = ("helvetica", 12, "italic", "bold"))
        self.getlist.tag_add("tagi", 0.0) # readme內容
        self.getlist.tag_config("tagi", foreground = "dark slate blue", \
                                background = "snow", \
                                font = ("Microsoft JhengHei UI", 12, "bold"))
        
        # 當Text元件變成可視時觸發事件，將內容還原成預設值
        s1 = "請依序輸入參數設定\n例：\n{\"trading_tax\": 0.0, \n\"trading_fee\": 0.001, \n...}"
        self.getlist.bind("<Visibility>", \
                          lambda e: TextReset(self.getlist, s1, "tagi"))
        
        self.btn_save_in = ttk.Button(self, text = "儲存自訂參數組", \
                                      command = self.saveinf)
        self.btn_save_in.place(relx = 0.28, rely = 0.66, \
                               relwidth = 0.2, relheight = 0.06)
        
        ### 訓練參數組選擇
        self.pset_ttl = ttk.Label(self, text = "訓練參數組選擇")
        self.pset_ttl.place(relx = 0.06, rely = 0.74, \
                            relwidth = 0.2, relheight = 0.06)
        # 自訂參數組
        self.btn_open_in = ttk.Button(self, text = "載入自訂參數組", \
                                      command = self.openinf)
        self.btn_open_in.place(relx = 0.28, rely = 0.74, \
                               relwidth = 0.2, relheight = 0.06)
        # 預設參數組
        self.dfok = tk.IntVar()
        self.dfok.set(0)
        self.dfpset = ttk.Checkbutton(self, text = "使用預設參數組", \
                                      variable = self.dfok, \
                                      onvalue = 1, offvalue = 0, \
                                      command = self.loaddfvals)
        self.dfpset.place(relx = 0.06, rely = 0.82, \
                          relwidth = 0.2, relheight = 0.06)
        
        # 確認/刷新
        # 在這一步時才會把載入&編輯好的參數傳入程式的全域變數
        self.btn_pset_ok = ttk.Button(self, text = "參數組設定確認", \
                                      command = self.updatepset)
        self.btn_pset_ok.place(relx = 0.28, rely = 0.82, \
                               relwidth = 0.2, relheight = 0.06)
        
        # 訓練模型還需要兩個輸入參數：資料數量與模型命名
        ### 資料組載入
        self.cc = tk.IntVar()
        self.cc.set(10)
        self.dps_ttl = ttk.Label(self, text = "訓練資料載入\n數據筆數：")
        self.dps_ttl.place(relx = 0.52, rely = 0.08, \
                           relwidth = 0.2, relheight = 0.1)
        self.dps = ttk.Entry(self, textvariable = self.cc)
        self.dps.place(relx = 0.74, rely = 0.08, \
                        relwidth = 0.2, relheight = 0.06)
        self.dpsmax = len(myWorld.GetFpathList("train"))
        footnote = "上限：" + str(self.dpsmax) + "筆資料"
        self.dps_ft = ttk.Label(self, text = footnote)
        self.dps_ft.place(relx = 0.74, rely = 0.16, \
                          relwidth = 0.2, relheight = 0.06)
        
        ### 訓練模型
        self.mm = tk.StringVar()
        self.mm.set(self.model_name)
        self.mm_ttl = ttk.Label(self, text = "設定AI模型存檔名稱")
        self.mm_ttl.place(relx = 0.52, rely = 0.26, \
                          relwidth = 0.2, relheight = 0.06)
        self.mm_ent = ttk.Entry(self, textvariable = self.mm)
        self.mm_ent.place(relx = 0.74, rely = 0.26, \
                          relwidth = 0.2, relheight = 0.06)
        
        self.fun_handler = self.train_rl # th_run 目標函式
        self.btn_train = ttk.Button(self, text = "訓練模型", \
                                    command = self.th_run)
        self.btn_train.place(relx = 0.74, rely = 0.34, \
                             relwidth = 0.2, relheight = 0.06)
        self.btn_train["state"] = "disabled"
        
        ### 訓練結果檢視
        self.btn_visdnn = ttk.Button(self, text = "模型架構檢視", \
                                     command = self.dnn_vis)
        self.btn_visdnn.place(relx = 0.74, rely = 0.42, \
                              relwidth = 0.2, relheight = 0.06)
        
        self.btn_visloss = ttk.Button(self, text = "訓練結果檢視", \
                                      command = self.train_vis)
        self.btn_visloss.place(relx = 0.52, rely = 0.42, \
                               relwidth = 0.2, relheight = 0.06)
        self.btn_visdnn["state"] = "disabled"
        self.btn_visloss["state"] = "disabled"
        # 畫布建立：將mpl作圖嵌入 (見self.train_vis)
        self.rx = 0.52
        self.ry = 0.50
        self.rw = 0.42
        self.rh = 0.38
        # self.s.set("就緒") # 結束
    
    def _create_type3(self, master):
        """ 建立第四個頁面內容，這裡處理模型驗證測試相關功能 """
        # 參數初始化
        self.inpset_dict = {} # 載入的參數組
        self.model_name = "demo_test" # RL agent 模型名稱 (副檔名為.h5)
        self.tts = 0. # train-test split fraction
        self.is_saved = True # 保留測試結果(檔案量可能非常大)
        self.ope_folder = myEnv.Trade_Path # 操作之資料夾
        
        self.status_ttl = ttk.Label(self, text = " 狀態 ")
        self.status_ttl.place(relx = 0.03, rely = 0.9, \
                              relwidth = 0.06, relheight = 0.06)
        self.s = tk.StringVar()
        self.s.set("就緒")
        self.status = ttk.Label(self, textvariable = self.s, \
                                relief = "sunken", background = "ivory", \
                                foreground = "dark slate blue", \
                                padding = 5, \
                                font = ("Microsoft JhengHei UI", 12, \
                                        "bold", "italic"))
        self.status.place(relx = 0.09, rely = 0.9, \
                          relwidth = 0.85, relheight = 0.06)
        self.status.bind("<Visibility>", \
                         lambda e: TSVReset(self.s, "就緒"))
        
        # 參數組檢視
        self.getlist_ttl = ttk.Label(self, text = "參數組檢視")
        self.getlist_ttl.place(relx = 0.06, rely = 0.08, \
                               relwidth = 0.42, relheight = 0.06)
        self.getlist = tk.Text(self, foreground = "dark slate blue", \
                               background = "snow", \
                               font = ("Microsoft JhengHei UI", 12, "bold"))
        self.getlist.place(relx = 0.06, rely = 0.16, \
                           relwidth = 0.42, relheight = 0.48)
        self.vscroll = ttk.Scrollbar(self.getlist, cursor = "arrow")
        self.vscroll.pack(side = "right", fill = 'y')
        self.getlist["yscrollcommand"] = self.vscroll.set
        self.vscroll["command"] = self.getlist.yview
        self.getlist.bind("<Visibility>", self.showpset)
        # self.getlist["state"] = "disabled"
        
        ### 測試資料選擇
        self.cc = tk.IntVar()
        self.cc.set(10)
        self.dps_ttl = ttk.Label(self, text = "測試資料載入\n數據筆數：")
        self.dps_ttl.place(relx = 0.52, rely = 0.08, \
                           relwidth = 0.2, relheight = 0.14)
        self.dps = ttk.Entry(self, textvariable = self.cc)
        self.dps.place(relx = 0.74, rely = 0.08, \
                       relwidth = 0.2, relheight = 0.06)
        self.dpsmax = len(myWorld.GetFpathList("test"))
        footnote = "上限：" + str(self.dpsmax) + "筆資料"
        self.dps_ft = ttk.Label(self, text = footnote)
        self.dps_ft.place(relx = 0.74, rely = 0.16, \
                          relwidth = 0.2, relheight = 0.06)
        
        ### 模型計算
        self.mm = tk.StringVar()
        self.mm.set(self.model_name)
        self.mm_ttl = ttk.Label(self, text = "設定AI模型讀檔名稱")
        self.mm_ttl.place(relx = 0.52, rely = 0.26, \
                          relwidth = 0.2, relheight = 0.06)
        self.mm_ent = ttk.Entry(self, textvariable = self.mm)
        self.mm_ent.place(relx = 0.74, rely = 0.26, \
                          relwidth = 0.2, relheight = 0.06)
        
        self.fun_handler = self.test_rl # th_run 目標函式
        self.btn_test = ttk.Button(self, text = "模型計算", \
                                   command = self.th_run)
        self.btn_test.place(relx = 0.52, rely = 0.34, \
                            relwidth = 0.2, relheight = 0.06)
        
        self.ttsok = tk.IntVar()
        self.ttsok.set(1)
        self.tts_chk = ttk.Checkbutton(self, text = "不對資料進行\n訓練-測試分割", \
                                       variable = self.ttsok, \
                                       onvalue = 1, offvalue = 0, \
                                       command = self.set_test_tts)
        self.tts_chk.place(relx = 0.74, rely = 0.42, \
                           relwidth = 0.2, relheight = 0.1)
        self.sav_ok = tk.IntVar()
        self.sav_ok.set(1)
        self.sav_chk = ttk.Checkbutton(self, text = "測試結果存檔\n(用於績效評估)", \
                                       variable = self.sav_ok, \
                                       onvalue = 1, offvalue = 0, \
                                       command = self.set_test_save)
        self.sav_chk.place(relx = 0.52, rely = 0.42, \
                           relwidth = 0.2, relheight = 0.10)
        
        ### 績效評估
        self.btn_visperf = ttk.Button(self, text = "績效評估", \
                                      command = self.test_perf)
        self.btn_visperf.place(relx = 0.63, rely = 0.56, \
                               relwidth = 0.2, relheight = 0.06)
        self.btn_clrperf = ttk.Button(self, text = "刪除測試結果", \
                                      command = self.clear_folder)
        self.btn_clrperf.place(relx = 0.63, rely = 0.64, \
                               relwidth = 0.2, relheight = 0.06)
        self.btn_visperf["state"] = "disabled"
        self.btn_clrperf["state"] = "disabled"
        # self.s.set("就緒") # 結束
    
    def _create_type4(self, master):
        """ 建立第五個頁面內容，這裡處理個別標的結果顯示功能 """
        self.infp = []
        self.ti_m = tk.StringVar()
        self.ti_m.set('')
        self.ti_s1 = tk.StringVar()
        self.ti_s1.set('')
        self.ti_s2 = tk.StringVar()
        self.ti_s2.set('')
        self.model_name = "demo_test" # RL agent 模型名稱 (副檔名為.h5)
        self.ope_folder = myEnv.Trade_Path # 操作之資料夾
        
        ### 單一標的選擇
        self.fp = tk.StringVar()
        self.fp.set('')
        self.fp_ttl = ttk.Label(self, text = "資料完整路徑")
        self.fp_ttl.place(relx = 0.06, rely = 0.08, \
                          relwidth = 0.2, relheight = 0.06)
        self.fp_ent = ttk.Entry(self, textvariable = self.fp)
        self.fp_ent.place(relx = 0.28, rely = 0.08, \
                          relwidth = 0.3, relheight = 0.06)
        self.btn_fp = ttk.Button(self, text = "檔案載入", \
                                 command = self.update_infp)
        self.btn_fp.place(relx = 0.38, rely = 0.16, \
                          relwidth = 0.2, relheight = 0.06)
        
        ### 技術指標選擇
        self.ti_ttl = ttk.Label(self, text = "繪製K線圖-技術指標選擇")
        self.ti_ttl.place(relx = 0.09, rely = 0.26, \
                          relwidth = 0.4, relheight = 0.06)
        # 主圖指標選擇
        self.m_ttl = ttk.Label(self, text = "主圖")
        self.m_ttl.place(relx = 0.09, rely = 0.34, \
                         relwidth = 0.12, relheight = 0.06)
        self.m_rb0 = ttk.Radiobutton(self, text = "不顯示", \
                                     variable = self.ti_m, \
                                     value = '')
        self.m_rb1 = ttk.Radiobutton(self, text = "SMA", \
                                     variable = self.ti_m, \
                                     value = "SMA")
        self.m_rb2 = ttk.Radiobutton(self, text = "VWMA", \
                                     variable = self.ti_m, \
                                     value = "VWMA")
        self.m_rb0.place(relx = 0.09, rely = 0.42, \
                         relwidth = 0.12, relheight = 0.06)
        self.m_rb1.place(relx = 0.09, rely = 0.5, \
                         relwidth = 0.12, relheight = 0.06)
        self.m_rb2.place(relx = 0.09, rely = 0.58, \
                         relwidth = 0.12, relheight = 0.06)
        # 副圖1指標選擇
        self.s1_ttl = ttk.Label(self, text = "副圖一")
        self.s1_ttl.place(relx = 0.23, rely = 0.34, \
                          relwidth = 0.12, relheight = 0.06)
        self.s1_rb0 = ttk.Radiobutton(self, text = "不顯示", \
                                      variable = self.ti_s1, \
                                      value = '')
        self.s1_rb1 = ttk.Radiobutton(self, text = "KD", \
                                      variable = self.ti_s1, \
                                      value = "KD")
        self.s1_rb2 = ttk.Radiobutton(self, text = "MACD", \
                                      variable = self.ti_s1, \
                                      value = "MACD")
        self.s1_rb3 = ttk.Radiobutton(self, text = "STD", \
                                      variable = self.ti_s1, \
                                      value = "STD")
        self.s1_rb4 = ttk.Radiobutton(self, text = "VIX", \
                                      variable = self.ti_s1, \
                                      value = "VIX")
        self.s1_rb5 = ttk.Radiobutton(self, text = "CVOL", \
                                      variable = self.ti_s1, \
                                      value = "CVOL")
        self.s1_rb0.place(relx = 0.23, rely = 0.42, \
                          relwidth = 0.12, relheight = 0.06)
        self.s1_rb1.place(relx = 0.23, rely = 0.5, \
                          relwidth = 0.12, relheight = 0.06)
        self.s1_rb2.place(relx = 0.23, rely = 0.58, \
                          relwidth = 0.12, relheight = 0.06)
        self.s1_rb3.place(relx = 0.23, rely = 0.66, \
                          relwidth = 0.12, relheight = 0.06)
        self.s1_rb4.place(relx = 0.23, rely = 0.74, \
                          relwidth = 0.12, relheight = 0.06)
        self.s1_rb5.place(relx = 0.23, rely = 0.82, \
                          relwidth = 0.12, relheight = 0.06)
        # 副圖2指標選擇
        self.s2_ttl = ttk.Label(self, text = "副圖二")
        self.s2_ttl.place(relx = 0.37, rely = 0.34, \
                          relwidth = 0.12, relheight = 0.06)
        self.s2_rb0 = ttk.Radiobutton(self, text = "不顯示", \
                                      variable = self.ti_s2, \
                                      value = '')
        self.s2_rb1 = ttk.Radiobutton(self, text = "KD", \
                                      variable = self.ti_s2, \
                                      value = "KD")
        self.s2_rb2 = ttk.Radiobutton(self, text = "MACD", \
                                      variable = self.ti_s2, \
                                      value = "MACD")
        self.s2_rb3 = ttk.Radiobutton(self, text = "STD", \
                                      variable = self.ti_s2, \
                                      value = "STD")
        self.s2_rb4 = ttk.Radiobutton(self, text = "VIX", \
                                      variable = self.ti_s2, \
                                      value = "VIX")
        self.s2_rb5 = ttk.Radiobutton(self, text = "CVOL", \
                                      variable = self.ti_s2, \
                                      value = "CVOL")
        self.s2_rb0.place(relx = 0.37, rely = 0.42, \
                          relwidth = 0.12, relheight = 0.06)
        self.s2_rb1.place(relx = 0.37, rely = 0.5, \
                          relwidth = 0.12, relheight = 0.06)
        self.s2_rb2.place(relx = 0.37, rely = 0.58, \
                          relwidth = 0.12, relheight = 0.06)
        self.s2_rb3.place(relx = 0.37, rely = 0.66, \
                          relwidth = 0.12, relheight = 0.06)
        self.s2_rb4.place(relx = 0.37, rely = 0.74, \
                          relwidth = 0.12, relheight = 0.06)
        self.s2_rb5.place(relx = 0.37, rely = 0.82, \
                          relwidth = 0.12, relheight = 0.06)
        
        ### 顯示K線圖
        self.btn_k = ttk.Button(self, text = "K線圖", \
                                command = self.show_k)
        self.btn_k.place(relx = 0.29, rely = 0.9, \
                         relwidth = 0.2, relheight = 0.06)
        self.btn_k["state"] = "disabled"
        
        ### 交易點位進出
        self.mm = tk.StringVar()
        self.mm.set(self.model_name)
        self.mm_ttl = ttk.Label(self, text = "設定AI模型讀檔名稱")
        self.mm_ttl.place(relx = 0.62, rely = 0.08, \
                          relwidth = 0.2, relheight = 0.06)
        self.mm_ent = ttk.Entry(self, textvariable = self.mm)
        self.mm_ent.place(relx = 0.62, rely = 0.16, \
                          relwidth = 0.2, relheight = 0.06)
        
        self.test_1_ttl = ttk.Label(self, text = "計算將會顯\n示交易點位", \
                                    justify = "center")
        self.test_1_ttl.place(relx = 0.84, rely = 0.1, \
                              relwidth = 0.1, relheight = 0.1)
        
        # self.fun_handler = self.test_rl_1 # th_run 目標函式
        # self.btn_test_1 = ttk.Button(self, text = "模型計算", \
        #                              command = self.th_run)
        # 避免多執行緒下mpl模組報錯：直接呼叫test_rl_1函式
        self.btn_test_1 = ttk.Button(self, text = "模型計算", \
                                     command = self.test_rl_1)
        self.btn_test_1.place(relx = 0.74, rely = 0.24, \
                              relwidth = 0.2, relheight = 0.06)
        self.btn_test_1["state"] = "disabled"
        
        ### 資產水位圖
        self.btn_visasset = ttk.Button(self, text = "資產水位圖", \
                                       command = self.asset_vis)
        self.btn_visasset.place(relx = 0.74, rely = 0.34, \
                                relwidth = 0.2, relheight = 0.06)
        self.btn_visasset["state"] = "disabled"
        # 畫布建立：將mpl作圖嵌入 (見self.asset_vis)
        self.rx = 0.62
        self.ry = 0.44
        self.rw = 0.32
        self.rh = 0.44
    
    # --- 類別內輔助與回調(callback)函式 --- #
    def saveinf(self):
        """ 呼叫儲存新檔對話方塊 """
        fpath = tk.filedialog.asksaveasfilename(title = "另存新檔", 
                                                filetypes = [("自定義輸入檔", "*.in"), 
                                                             ("所有檔案", "*")], 
                                                initialdir = self.save_path, 
                                                initialfile = self.default_infname)
        self.s.set("存入檔案：" + fpath)
        # 寫入檔案
        try:
            if not fpath is None and fpath != '': # 有確實存檔取得路徑
                self.infploaded = fpath
                self.infcontent = self.getlist.get(0.0, tk.END)
                with open(fpath, 'w', encoding = "utf-8") as f:
                    f.write(self.infcontent)
                try:
                    # 若btn_scraper_go有定義就改設定，沒有就pass
                    self.btn_scraper_go["state"] = "normal"
                except Exception:
                    pass
        except Exception as e:
            self.getlist.insert(tk.END, "Fail to save: \n" + fpath, "tage")
            self.getlist.insert(tk.END, e, "tage")
        # 更新檔案清單 (追蹤紀錄，目前未使用)
        rawflist = os.listdir(self.save_path)
        self.infplist = [self.save_path + '\\' + ff for ff in rawflist \
                         if os.path.splitext(ff)[-1].lower() == ".in"]
        self.infnlist = [os.path.splitext(ff)[0] for ff in rawflist \
                         if os.path.splitext(ff)[-1].lower() == ".in"]
    
    def openinf(self):
        """ 呼叫開啟舊檔對話方塊，並將內容載入Text元件中 """
        fpath = tk.filedialog.askopenfilename(title = "開啟舊檔", 
                                              filetypes = [("自定義輸入檔", "*.in"), 
                                                           ("所有檔案", "*")], 
                                              initialdir = self.open_path, 
                                              initialfile = self.default_infname)
        self.s.set("開啟檔案：" + fpath)
        try:
            if not fpath is None and fpath != '': # 有確實開啟舊檔取得路徑
                self.getlist.delete(0.0, tk.END)
                f = open(fpath, 'r', encoding = "utf-8")
                self.getlist.insert(tk.END, f.read(), "tagi")
                self.infploaded = fpath
                self.infcontent = self.getlist.get(0.0, tk.END)
                try:
                # 若btn_scraper_go有定義就改設定，沒有就pass
                    self.btn_scraper_go["state"] = "normal"
                except Exception:
                    pass
            # 取消開啟檔案則跳過以上動作
        except Exception as e:
            self.getlist.insert(tk.END, "Fail to open: \n" + fpath, "tage")
            self.getlist.insert(tk.END, e, "tage")
    
    def disable_ent_ymd(self):
        """ 按鍵功能鎖定 """
        if self.d.get() is True: # Checked
            self.b.set(str(dt.date.today()))
            self.c.set(myParams.DaysPerYear)
            self.ymd["state"] = "disabled"
            self.days["state"] = "disabled"
        else:
            self.ymd["state"] = "normal"
            self.days["state"] = "normal"
            self.b.set(str(dt.date.today()))
            self.c.set(180)
    
    def scraper(self):
        """ 呼叫myScraper模組主函式，運行網路爬蟲擷取資料 """
        t1 = dt.datetime.now()
        try:
            inf = self.infploaded
            datasrc = self.a.get()
            ed = dt.datetime.fromisoformat(self.b.get())
            dayspan = self.c.get()
            isyfdd = False # 是否使用預設日期(Yahoo Finance)
            if self.d.get() is True: # Overwrite date info
                ed = dt.date.today()
                dayspan = myParams.DaysPerYear
                isyfdd = True
            self.s.set("擷取資料中，使用自選清單：" + inf)
            myScraper.Main(inf, ed, dayspan, datasrc, isyfdd)
        except Exception as e:
            tk.messagebox.showerror("網路擷取資料失敗！錯誤訊息：\n", e)
            self.s.set("擷取資料失敗！")
        else: # 資料後處理
            try:
                kw_list = list(myParams.DefinedTechInd) # Calculate all technical indicators
                if datasrc == mySet.PriceDataSrc[0]: # 證交所
                    self.s.set("數據後處理...證交所資料格式解析")
                    myPostP.Main(TD, kw_list, \
                                 myParams.Src_Dict[myParams.Src_List[0]])
                    myPostP.Main(TD, kw_list, \
                                 myParams.Src_Dict[myParams.Src_List[1]])
                elif datasrc == mySet.PriceDataSrc[1]: # Yahoo finance
                    self.s.set("數據後處理...雅虎財經資料格式解析")
                    myPostP.Main(TD, kw_list, \
                                 myParams.Src_Dict[myParams.Src_List[2]])
                else:
                    raise Exception("無效的資料來源： " + str(datasrc))
            except Exception as e:
                tk.messagebox.showerror("資料後處理失敗！錯誤訊息：\n", e)
                self.s.set("資料後處理失敗！")
            else:
                t2 = dt.datetime.now()
                delta_t = str(t2 - t1)
                self.s.set("資料更新成功！用時：" + delta_t)
    
    def loaddfvals(self):
        """ 載入預設資料，並顯示於文字編輯器上 """
        if self.dfok.get() == 1:
            self.s.set("載入預設參數組：" + self.default_psetname)
            try:
                self.getlist.delete(0.0, tk.END)
                # 回傳字典類型的參數組(引數isreturndict = True)
                self.dfpset_dict = \
                    myWorld.ChangeTheWorld(self.default_psetname, True)
                self.inpset = self.dfpset_dict
                counts = 0
                self.getlist.insert(tk.END, "{")
                for k, v in self.dfpset_dict.items():
                    if not type(v) is str: # int, float, bool, list, set, ...
                        ss = "'{k}': {v}".format(k=k, v=v)
                    else: # add quotation mark to dict value
                        ss = "'{k}': '{v}'".format(k=k, v=v)
                    self.getlist.insert(tk.END, ss)
                    counts += 1
                    if counts < len(self.dfpset_dict):
                        self.getlist.insert(tk.END, ", \n")
                    else:
                        self.getlist.insert(tk.END, "}")
            except Exception as e:
                self.getlist.insert(tk.END, "Fail to load param set = " + \
                                    self.default_psetname)
                self.getlist.insert(tk.END, e)
    
    def updatepset(self):
        """ 刷新參數組內容 """
        # 預期getlistcontent包含所有全域參數的鍵值對
        global GetListContent
        getlistcontent = self.getlist.get(0.0, tk.END)
        GetListContent = getlistcontent
        if getlistcontent is None:
            getlistcontent = '{}'
        try:
            # 將文字格式轉換成字典類型，並以**操作子unpack作為函式引數
            # eval方法可自動清除換行符號'\n'
            self.inpset_dict = \
                myWorld.ChangeTheWorld_V2(**eval(getlistcontent))
            self.getlist.delete(0.0, tk.END)
            # 若getlistcontent未包含全部鍵值對，則以下將顯示
            counts = 0
            self.getlist.insert(tk.END, "{")
            for k, v in self.inpset_dict.items():
                if not type(v) is str:
                    ss = "'{k}': {v}".format(k=k, v=v)
                else:
                    ss = "'{k}': '{v}'".format(k=k, v=v)
                self.getlist.insert(tk.END, ss)
                counts += 1
                if counts < len(self.inpset_dict):
                    self.getlist.insert(tk.END, ", \n")
                else:
                    self.getlist.insert(tk.END, "}")
        except Exception:
            self.s.set("更新參數組出現格式錯誤！")
        else:
            try:
                self.dfok.set(0)
            except Exception:
                pass
            try:
                self.btn_train["state"] = "normal"
            except Exception:
                pass
            self.s.set("更新參數組成功！")
    
    def train_rl(self):
        """ 進行RL模型訓練 """
        t1 = dt.datetime.now()
        if not self.mm.get() is None \
            and not self.mm.get() == '':
            self.model_name = self.mm.get()
        self.s.set("模型訓練中...模型名稱：" + self.model_name)
        try:
            myEnv.GetColNames(td = TD, kwlist = KWList)
            self.results_df = myWorld.Training(filename = self.cc.get(), 
                                               symbol = self.model_name, 
                                               save_results = True, 
                                               israndompick = True, 
                                               agent_name = myAgent.Chosen_Agent, 
                                               episodes = myWorld.Episodes, 
                                               episode_length = myWorld.Episode_Length, 
                                               train_test_split = myWorld.Train_Test_Split, 
                                               history_length = myWorld.History_Length, 
                                               time_fee = myWorld.Time_Fee, 
                                               trading_fee = myWorld.Trading_Fee, 
                                               trading_tax = myWorld.Trading_Tax, 
                                               memory_size = myAgent.Memory_Size, 
                                               batch_size = myAgent.Batch_Size)
        except Exception:
            self.s.set("運算過程出現錯誤！")
        else:
            try:
                self.btn_visloss["state"] = "normal"
                self.btn_visdnn["state"] = "normal"
            except Exception:
                pass
            t2 = dt.datetime.now()
            delta_t = str(t2 - t1)
            self.s.set("模型訓練完成！用時：" + delta_t)
    
    def th_run(self):
        """ 使用多執行緒，fun_handler為一函式物件需被定義 """
        try:
            th = threading.Thread(target = self.fun_handler)
            th.setDaemon(True) # 守護執行緒
            th.start()
        except RuntimeError as e:
            tk.messagebox.showerror("執行緒錯誤", e)
            msg = "若為'main thread is not in main loop'，\n請關閉所有相關視窗後重新開啟..."
            tk.messagebox.showinfo("處理建議", msg)
        except Exception as e:
            tk.messagebox.showerror("執行緒不明錯誤", e)
    
    def canvas_draw(self, rx, ry, rw, rh):
        """ 將mpl作圖嵌入tk之Canvas元件中 """
        try:
            self.canvas = FigureCanvasTkAgg(self.fig, master = self)
            self.canvas.draw()
            self.canvas.get_tk_widget().place(relx = rx, 
                                              rely = ry, 
                                              relwidth = rw, 
                                              relheight = rh)
        except Exception as e:
            print("Exception occurs in canvas_draw method:")
            print(e)
            pass
    
    def train_vis(self):
        """ 訓練結果作圖檢視 """
        try:
            self.fig, self.axs = \
                myAna.Training_Visualizer(self.results_df, True)
        except Exception as e:
            self.s.set("訓練結果作圖過程出現錯誤！")
            print(e)
        else:
            try:
                self.canvas_draw(self.rx, self.ry, self.rw, self.rh)
            except Exception as e:
                self.s.set("作圖嵌入出現錯誤！另開視窗作圖...")
                print(e)
                myAna.Training_Visualizer(self.results_df, False)
            else:
                self.s.set("顯示訓練結果")
    
    def dnn_vis(self):
        """ 模型架構檢視與圖檔輸出 """
        try:
            mpath = RefPath + '\\' + myParams.AI_Foldername
            aname = self.model_name + '_' + myAgent.Chosen_Agent + '.h5'
            fpath = mpath + '\\' + aname
            dnn = myDNNVisualizer.load_model(fpath)
            """
            pydot_path = mpath + '\\' + aname + '.png'
            myDNNVisualizer.Export_Model_Struct(dnn, 
                                                to_file = pydot_path, 
                                                show_shapes = True, 
                                                show_layer_names = True)
            """
            vis_path = mpath + '\\' + aname + "_network.png"
            ins = dnn.get_weights()[0].shape[0]
            outs = dnn.get_weights()[-1].shape[0]
            innames = ["In-" + str(i+1) for i in range(ins)]
            outnames = ["Out-" + str(i+1) for i in range(outs)]
            dnncoef = dnn.get_weights()
            myDNNVisualizer.VisualizeDNN(dnncoef, 
                                         innames, 
                                         outnames, 
                                         vis_path)
        except Exception:
            self.s.set("顯示模型架構過程出現錯誤")
        else:
            self.s.set("顯示模型架構")
    
    def showpset(self, evt):
        """ 顯示參數組內容 """
        try:
            # 不提供引數，即取得當下參數鍵值對
            self.inpset_dict = \
                myWorld.ChangeTheWorld_V2(**eval(GetListContent))
            self.getlist.delete(0.0, tk.END)
            counts = 0
            self.getlist.insert(tk.END, "{")
            for k, v in self.inpset_dict.items():
                if not type(v) is str:
                    ss = "'{k}': {v}".format(k=k, v=v)
                else:
                    ss = "'{k}': '{v}'".format(k=k, v=v)
                self.getlist.insert(tk.END, ss)
                counts += 1
                if counts < len(self.inpset_dict):
                    self.getlist.insert(tk.END, ", \n")
                else:
                    self.getlist.insert(tk.END, "}")
        except Exception:
            self.s.set("顯示參數組出現錯誤！")
    
    def set_test_tts(self):
        """ 設定 Train_Test_Split """
        if self.ttsok.get() == 1: # ok
            self.tts = 0.
        else:
            self.tts = myWorld.Train_Test_Split
    
    def set_test_save(self):
        """ 設定是否存檔 """
        if self.sav_ok.get() == 1: # ok
            self.is_saved = True
        else:
            self.is_saved = False
    
    def test_rl(self):
        """ 進行RL模型測試驗證 """
        t1 = dt.datetime.now()
        if not self.mm.get() is None \
            and not self.mm.get() == '':
            self.model_name = self.mm.get()
        self.s.set("模型測試中...模型名稱：" + self.model_name)
        try:
            myEnv.GetColNames(td = TD, kwlist = KWList)
            self.results_df_dict = myWorld.Testing(filename = self.cc.get(), 
                                                   symbol = self.model_name, 
                                                   search_folder = "test", 
                                                   train_test_split = self.tts, 
                                                   save_results = self.is_saved, 
                                                   render_save = False, 
                                                   render_show = False, 
                                                   israndompick = True, 
                                                   agent_name = myAgent.Chosen_Agent, 
                                                   history_length = myWorld.History_Length, 
                                                   time_fee = myWorld.Time_Fee, 
                                                   trading_fee = myWorld.Trading_Fee, 
                                                   trading_tax = myWorld.Trading_Tax)
        except Exception:
            self.s.set("計算過程出現錯誤！")
        else:
            try:
                self.btn_visperf["state"] = "normal"
                self.btn_clrperf["state"] = "normal"
            except Exception:
                pass
            t2 = dt.datetime.now()
            delta_t = str(t2 - t1)
            self.s.set("模型計算測試完成！用時：" + delta_t)
    
    def test_perf(self):
        """
        批次績效評估，並顯示Excel檔
        由於計算檔案是隨機調用的，因此無法用輸入檔建立檔名清單
        而必須從輸出檔建立清單(並排除檔名重複)
        """
        fnamelist = []
        test_str = ["reward", 
                    "act_policy", 
                    "calc_return", 
                    "asset_rec", 
                    "trade_rec"]
        try:
            rawfnlist = os.listdir(self.ope_folder)
            # print("Search path: ", self.ope_folder)
            # print("Show raw fname list: \n", rawfnlist)
            for fn in rawfnlist:
                fnext = os.path.splitext(fn)[-1]
                for tstr in test_str:
                    pattern = "[_](?=" + tstr + ')'
                    if len(re.split(pattern, fn)) == 2: 
                        fname = re.split(pattern, fn)[0]
                        # print("fn = ", fn)
                        # print(re.split(pattern, fn))
                        # print("tstr = ", tstr)
                        # print("fname = ", fname)
                        if not fname in fnamelist and fnext.lower() == '.csv':
                            # 找到.csv檔案，同時檔名是第一次出現的
                            fnamelist.append(fname)
        except Exception as e:
            print(e)
            self.s.set("建立待績效分析清單時出現錯誤！")
        else:
            print("Show file name list: \n", fnamelist)
            try:
                perf_df = myWorld.Batch_Describe_Performance(fnamelist)
            except Exception as e:
                print(e)
                self.s.set("績效分析計算時出現錯誤！")
            else:
                print("Show results: \n", perf_df)
                n = str(len(fnamelist))
                self.s.set("績效分析計算完成！資料數：" + n)
                tar_f = myWorld.RefPath + '\\' + "samamind_performance.xlsx"
                OpenFile(tar_f) # 用Excel開啟檔案
    
    def clear_folder(self):
        """ 清空模型計算結果 """
        rm_folder = self.ope_folder # removed root folder
        msgbox = \
            tk.messagebox.askquestion("資料夾清空", 
                                      "請確認是否清空：\n" + rm_folder)
        if msgbox == "yes":
            if os.path.isdir(rm_folder):
                rmtree(rm_folder, ignore_errors = True)
                sleep(2)
                tk.messagebox.showinfo("完成", "資料夾已清空！")
            else:
                sss = "以下資料夾不存在: " + rm_folder
                tk.messagebox.showinfo("嘗試失敗", sss)
        else:
            tk.messagebox.showinfo("完成", "取消清空資料夾！")
        if not os.path.isdir(rm_folder):
            os.mkdir(rm_folder)
    
    def update_infp(self):
        """ 更新self.infp """
        fpath = tk.filedialog.askopenfilename(title = "開啟舊檔", 
                                              filetypes = [("Excel檔案", "*.xlsx"), 
                                                           ("逗號分隔檔案", "*.csv"), 
                                                           ("所有檔案", "*")], 
                                              initialdir = myPostP.PostP_Path, 
                                              initialfile = '')
        if not fpath is None and fpath != '': # 有確實開啟舊檔取得路徑
            self.fp.set(fpath) # GUI輸出顯示
            # 將路徑的slash做修正：
            # fnext = re.findall(r'[^\\]+', fpath)[-1]
            # fnext = re.findall(r'[^/]+', fpath)[-1]
            # fpath_new = myPostP.PostP_Path + '\\' + fnext
            fpath_new = re.sub(r'/', r'\\', fpath) # 允許預設路徑以外的來源
            self.infp = [fpath_new] # 存入清單待用
            try:
                self.btn_k["state"] = "normal"
            except Exception:
                pass
            try:
                self.btn_test_1["state"] = "normal"
            except Exception:
                pass
    
    def show_k(self):
        """ 顯示K線圖 """
        # 覆寫全域參數
        # 目前僅做技術指標欄位和報酬率計算的修改，交易天期則維持不變
        tilist = list(myParams.DefinedTechInd)
        myPostP.GetSelectedTechInd(td = TD, kwlist = tilist)
        myTI.TISetUp(tilist, False)
        # 資料檔案格式整理
        df0 = pd.read_excel(self.infp[0], header = 0, \
                            index_col = None, skiprows = None)
        df = pd.DataFrame(data = [], columns = myTI.BaseColumns, \
                          index = df0["Date"].values)
        cc = list(df.columns)
        df[cc] = df0[cc].values
        df.index = pd.to_datetime(df.index)
        # 繪圖！
        dpset = myPostP.ABC
        ti_m = self.ti_m.get()
        ti_s1 = self.ti_s1.get()
        ti_s2 = self.ti_s2.get()
        fname = re.findall(r'[^\\]+(?=\.)', self.infp[0])[-1]
        # fname = re.findall(r'[^/]+(?=\.)', self.infp[0])[-1]
        ti_graph = myTI.TIGraph_Mpf(df, dpset, fig_size_x = 16, \
                                    fig_size_y = 12)
        if ti_m == '' and ti_s1 == '' and ti_s2 == '':
            ti_graph.show_basic(fname)
        else:
            kwarg = {}
            if not ti_m is None and ti_m != '':
                kwarg["ti_main"] = ti_m
                fname = fname + '_' + ti_m
            else:
                fname = fname + "_X"
            if not ti_s1 is None and ti_s1 != '':
                kwarg["ti_sub1"] = ti_s1
                fname = fname + '_' + ti_s1
            else:
                fname = fname + "_X"
            if not ti_s2 is None and ti_s2 != '':
                kwarg["ti_sub2"] = ti_s2
                fname = fname + '_' + ti_s2
            else:
                fname = fname + "_X"
            ti_graph.show_adv(fname, **kwarg)
        # 還原設定
        sleep(1)
        ti_graph.emptydfs()
        myPostP.GetSelectedTechInd(td = TD, kwlist = KWList)
        tilist = list(myPostP.SelectedTechInd)
        myTI.TISetUp(tilist, True)
    
    def test_rl_1(self):
        """ 進行單一標的的RL模型計算 """
        if not self.mm.get() is None \
            and not self.mm.get() == '':
            self.model_name = self.mm.get()
        try:
            myEnv.GetColNames(td = TD, kwlist = KWList)
            self.results_df_dict = myWorld.Testing(filename = self.infp[0], 
                                                   symbol = self.model_name, 
                                                   search_folder = "test", 
                                                   train_test_split = 0., 
                                                   save_results = True, 
                                                   render_save = True, 
                                                   render_show = True, 
                                                   israndompick = False, 
                                                   agent_name = myAgent.Chosen_Agent, 
                                                   history_length = myWorld.History_Length, 
                                                   time_fee = myWorld.Time_Fee, 
                                                   trading_fee = myWorld.Trading_Fee, 
                                                   trading_tax = myWorld.Trading_Tax)
        except Exception:
            pass
        else: 
            try: 
                self.btn_visasset["state"] = "normal"
            except Exception:
                pass
    
    def asset_vis(self):
        """ 檢視資產水位圖 """
        try:
            fn = re.findall(r'[^\\]+(?=\.)', self.infp[0])[-1]
            # fn = re.findall(r'[^/]+(?=\.)', self.infp[0])[-1]
            apath = self.ope_folder + '\\' + fn + "_asset_rec.csv"
            asset_df = pd.read_csv(apath, index_col = None, \
                                   delimiter = myParams.Delimiter, \
                                   skiprows = None, header = 0)
            try:
                self.fig, self.axs = \
                    myAna.Asset_Visualizer(asset_df, True)
                self.canvas_draw(self.rx, self.ry, self.rw, self.rh)
            except Exception as e:
                print(e)
                myAna.Asset_Visualizer(asset_df, False)
        except Exception as e:
            print(e)


class MainWindow:
    """ 主視窗類別，使用選單驅動 """
    def __init__(self):
        """ 主視窗初始化 """
        # GUI各頁面標題
        self._tab_names = ["系統功能", 
                           "自選清單資料更新", 
                           "模型訓練", 
                           "模型測試", 
                           "個別標的結果顯示"]
        # 建立視窗，tk物件實體化
        self.master = tk.Tk() # Toplevel widget
        self.master.protocol("WM_DELETE_WINDOW", self.quitloop)
        SetIcon(self.master) # 更換主視窗(與隨後出現的各訊息窗)的圖示
        self.master.title("SaMaMind Lite")
        # 視窗大小/組態/字型設定
        # self.master.configure(background = "seashell")
        self.master.geometry("{}x{}+{}+{}".format(Default_Width, 
                                                  Default_Height, 
                                                  Default_OffsetX, 
                                                  Default_OffsetY))
        
        # 頁面標籤建立
        self.panel = ttk.Notebook(self.master)
        
        # 設定外觀與字體
        style = ttk.Style(self.master)
        style.configure("TNotebook", 
                        background = "seashell", 
                        foreground = "sienna", 
                        font = ("Microsoft JhengHei UI", 16))
        style.configure("TNotebook.Tab", 
                        background = "alice blue", 
                        foreground = "dark slate blue", 
                        padding = [8, 2], 
                        font = ("Microsoft JhengHei UI", 12, "bold"))
        style.configure("TFrame", 
                        background = "alice blue", 
                        foreground = "dark slate blue", 
                        font = ("Microsoft JhengHei UI Light", 16))
        style.configure("TLabel", 
                        background = "azure", 
                        foreground = "cornflower blue", 
                        font = ("Microsoft JhengHei UI", 12, "bold"))
        style.configure("TButton", 
                        background = "azure", 
                        foreground = "cornflower blue", 
                        font = ("Microsoft JhengHei UI", 12, "bold"))
        style.configure("TRadiobutton", 
                        background = "azure", 
                        foreground = "cornflower blue", 
                        font = ("Microsoft JhengHei UI", 12, "bold"))
        style.configure("TCheckbutton", 
                        background = "azure", 
                        foreground = "cornflower blue", 
                        font = ("Microsoft JhengHei UI", 12, "bold"))
        style.configure("TEntry", 
                        background = "seashell", 
                        foreground = "sienna", 
                        font = ("Arial", 12, "bold"))
        self.tabs = []
        for i in range(len(self._tab_names)):
            self.tabs.append(TkTab(self.master, i))
            self.panel.add(self.tabs[i], \
                           text = self._tab_names[i])
        self.panel.select(self.tabs[0])
        # 透過 expand 打開 fill 屬性，允許各元件隨視窗大小變動大小
        self.panel.pack(expand = 1, fill = "both")
    
    def mainloop(self, isfixedsize = False):
        """ 執行主程式 """
        if isfixedsize: # 凍結視窗大小
            self.master.resizable(width = False, height = False)
            # Set minsize and maxsize to same LxW works as well:
            # self.master.minsize(Default_Width, Default_Height)
            # self.master.maxsize(Default_Width, Default_Height)
        self.master.mainloop()
    
    def quitloop(self):
        """ 離開主程式 """
        msgbox = tk.messagebox.askquestion("離開程式", "請確認是否離開？")
        if msgbox == "yes":
            self.master.destroy()

# --- --- #

if __name__ == "__main__":
    ## Icon圖檔測試 
    # root = tk.Tk()
    # SetIcon(root)
    # root.mainloop()
    # root.destroy()
    
    ## GUI測試
    window = MainWindow()
    window.mainloop()
    # window.mainloop(True) # 固定視窗尺寸，禁止縮放
    # window.quitloop()
# Done!
