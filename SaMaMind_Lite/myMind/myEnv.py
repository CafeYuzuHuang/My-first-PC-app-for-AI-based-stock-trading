# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn import preprocessing as prep
import matplotlib as mpl
import matplotlib.pyplot as plt

from myUtilities import myParams
from myUtilities import myLog
from myStockScraper import myPostProcessing as myPostP

"""
myEnv.py 所具有的功能：(基於Saeed腳本做大幅度改寫)
(1) 讀取股價與技術指標資料，並做 train-test split
(2) 使RL agent可與環境互動，以 Indicator_1.step()實現之
(3) 交易進出點位視覺化，並新增資產水位變化的視覺化

程式碼的修改方向，參考以下文獻：
[1] Thibaut Theate and Damien Ernest, "An Application of Deep Reinforcement 
Learning to Algorithmic Trading" (arXiv:2004.06627v3) (2020)
Python script source:
https://github.com/ThibautTheate/An-Application-of-Deep-Reinforcement-Learning-to-Algorithmic-Trading
"""

# Global variables and constants:
plt.set_loglevel("warning") # avoid log polluted by mpl

LogName = __name__ # 日誌名稱設為模組名稱
RefPath = '' # 參考路徑
if __name__ == "__main__": # executed
    RefPath = os.path.dirname(os.getcwd())
    LogName = None # 當此腳本被執行時為根日誌(不設名稱)
else: # imported
    modulepath = os.path.dirname(__file__)
    RefPath = os.path.dirname(modulepath)
Logger = myLog.Log(myParams.DefaultRootLogLevel, LogName)

Trade_Path = RefPath + '\\' + myParams.Trade_Foldername # 存放運算結果
if not os.path.isdir(Trade_Path):
    os.mkdir(Trade_Path)


Default_TechInd_List = list(myParams.DefinedTechInd)
Cols_1 = []
Cols_2 = []

# 是否使用成交量作為input，由於台股成交量容易受法規
# (處置股、現沖與融資券規定、以及日後的造市制度)和漲停板干擾，故不建議使用！
IsVolUsed = False
# IsVolUsed = True # for testing & comparison

# 滑價成本：流動性好的股票一般可以低到0.1%左右，但流動性差的可以價差好幾個tick
# 在此取台股tick差的最大比例，即100.5元的台股報價跳動一檔差0.5元，即0.5%
Default_Spread = 0.005 # 買賣價差 = 賣價 - 買價 = Ask - Bid > 0.
Default_Prep_IsReturnBasis = False # 前處理所用的資料。False: 使用價位; True: 使用報酬率
Default_Unit_Quantity = 1000 # 單位買量 = 1張 = 1000股
Default_Initial_Cash = 1000000 # 初始資金；不考慮槓桿操作故不設保證金下限
Default_Max_Underwater_Ratio = -0.5 # 可忍受最大虧損，預設為初始本金之1/2
Default_Keep_Cash_Ratio = 0.5 # 每次買進證券時預留現金的比例
Default_Hold_Period_Upper = myPostP.TradePeriod # 持有天數上限
# Default_Hold_Period_Upper = None # 不設定持有天期上限，即不對時間成本做懲罰

# 全域變數：以引數傳遞，用於TAStreamer
Spread = Default_Spread
Prep_IsReturnBasis = Default_Prep_IsReturnBasis
# 全域變數：直接於Indicator_1調用
Unit_Quantity = Default_Unit_Quantity
Initial_Cash = Default_Initial_Cash
Max_Underwater_Ratio = Default_Max_Underwater_Ratio
Keep_Cash_Ratio = Default_Keep_Cash_Ratio
Hold_Period_Upper = Default_Hold_Period_Upper

# --- --- #

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def EnvReset():
    """ 還原初始設定 """
    global Spread, Prep_IsReturnBasis, Unit_Quantity
    global Initial_Cash, Max_Underwater_Ratio, Keep_Cash_Ratio
    global Hold_Period_Upper
    Spread = Default_Spread
    Prep_IsReturnBasis = Default_Prep_IsReturnBasis
    Unit_Quantity = Default_Unit_Quantity
    Initial_Cash = Default_Initial_Cash
    Max_Underwater_Ratio = Default_Max_Underwater_Ratio
    Keep_Cash_Ratio = Default_Keep_Cash_Ratio
    Hold_Period_Upper = Default_Hold_Period_Upper
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def EnvSetUp(**kwargs):
    """
    設定RL運行環境，修改預設參數。已定義關鍵字：
    spread, (float, [0, ) )
    prep_isreturnbasis, (bool)
    unit_quantity, (int, [1, ) )
    initial_cash, (int, [1, ) )
    max_underwater_ratio, (float, [-1, 0])
    keep_cash_ratio, (float, [0, 1])
    hold_period_upper (int, [1, ), or None)
    """
    global Spread, Prep_IsReturnBasis, Unit_Quantity
    global Initial_Cash, Max_Underwater_Ratio, Keep_Cash_Ratio
    global Hold_Period_Upper
    EnvReset()
    
    if "spread" in kwargs:
        if type(kwargs["spread"]) is float and kwargs["spread"] >= 0:
            Spread = kwargs["spread"]
        else:
            Logger.warning("Invalid spread (required: > 0.0).")
    if "prep_isreturnbasis" in kwargs:
        if type(kwargs["prep_isreturnbasis"]) is bool:
            Prep_IsReturnBasis = kwargs["prep_isreturnbasis"]
        else:
            Logger.warning("Invalid prep_isreturnbasis (required: bool).")
    if "unit_quantity" in kwargs:
        if int(kwargs["unit_quantity"]) >= 1:
            Unit_Quantity = int(kwargs["unit_quantity"])
        else:
            Logger.warning("Invalid unit_quantity (required: >= 1).")
    if "initial_cash" in kwargs:
        if int(kwargs["initial_cash"]) >= 1:
            Initial_Cash = int(kwargs["initial_cash"])
        else:
            Logger.warning("Invalid initial_cash (required: >= 1).")
    if "max_underwater_ratio" in kwargs:
        if type(kwargs["max_underwater_ratio"]) is float and \
            kwargs["max_underwater_ratio"] <= 0 and \
            kwargs["max_underwater_ratio"] >= -1:
            Max_Underwater_Ratio = kwargs["max_underwater_ratio"]
        else:
            Logger.warning("Invalid max_underwater_ratio!")
            Logger.warning("(Required: >= -1.0 and <= 0.0)")
    if "keep_cash_ratio" in kwargs:
        if type(kwargs["keep_cash_ratio"]) is float and \
            kwargs["keep_cash_ratio"] <= 1 and \
            kwargs["keep_cash_ratio"] >= 0:
            Keep_Cash_Ratio = kwargs["keep_cash_ratio"]
        else:
            Logger.warning("Invalid keep_cash_ratio!")
            Logger.warning("(Required: >= 0.0 and <= 1.0)")
    if "hold_period_upper" in kwargs:
        if kwargs["hold_period_upper"] is None:
            Hold_Period_Upper = kwargs["hold_period_upper"]
        elif int(kwargs["hold_period_upper"]) >= 1:
            Hold_Period_Upper = int(kwargs["hold_period_upper"])
        else:
            Logger.warning("Invalid hold_period_upper!")
            Logger.warning("(Required: None or >= 1)")
    # Examine the variables:
    Logger.info("spread: %s" % str(Spread))
    Logger.info("prep_isreturnbasis: %s" % str(Prep_IsReturnBasis))
    Logger.info("unit_quantity: %s" % str(Unit_Quantity))
    Logger.info("initial_cash: %s" % str(Initial_Cash))
    Logger.info("max_underwater_ratio: %s" % str(Max_Underwater_Ratio))
    Logger.info("keep_cash_ratio: %s" % str(Keep_Cash_Ratio))
    Logger.info("hold_period_upper: %s" % str(Hold_Period_Upper))
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def GetColNames(td = 20, kwlist = None):
    """ 取得已定義的欄位名稱(呼叫myPostProcessing模組) """
    global Cols_1, Cols_2
    if kwlist is None: # Do not apply [] or {} as default argument(s)!
        kwlist = Default_TechInd_List
    myPostP.GetSelectedTechInd(td, kwlist)
    if not IsVolUsed:
        Logger.info("Caution: column 'V' and 'R-V' are not used!")
        ColV = ["V", "R-V"] # may include VWMA columns
        Cols_1 = [i for i in myPostP.Cols_Selected if not i in ColV]
        cols_tmp = myPostP.Cols_R_Included
        Cols_2 = [i for i in cols_tmp if (not i in Cols_1 and not i in ColV)]
    else:
        Logger.info("Column 'V' and 'R-V' are applied to the environment!")
        Cols_1 = myPostP.Cols_Selected # 價量/技術指標數值
        # 以下取得價格與技術指標之報酬率(一階差分)，以及昨量比
        Cols_2 = [i for i in myPostP.Cols_R_Included if not i in Cols_1]
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
# @myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def Penaltize_Time_Fee(time_fee, t_hold):
    """
    訓練 RL 模型時使用：使時間成本為持倉時間的函數，使用 relu function
    Penaltized time fee =
        time fee (if t_hold <= t_up_lim)
        time fee + fac * (t_hold - t_up_lim) (if t_hold > t_up_lim)
    """
    t_up_lim = Hold_Period_Upper
    rel_fac = 10.
    penaltized_time_fee = time_fee
    if t_hold > t_up_lim:
        fac = rel_fac * time_fee / t_up_lim
        penaltized_time_fee += fac * (t_hold - t_up_lim)
    return penaltized_time_fee

class DataGenerator:
    """
    Abstract class for a data generator. Do not use directly.
    Overwrite the _generator method to create a custom data generator.
    """
    def __init__(self, **gen_kwargs):
        """
        Initialization function. The API (gen_kwargs) should be defined in
        the function _generator.
        """
        self.gen_kwargs = gen_kwargs
        DataGenerator.rewind(self)
    
    # 以staticmethod方法修飾，表示不需要第一個引數(即self)
    @staticmethod
    def _generator(**kwargs):
        """
        Generator function. The keywords arguments entirely defines the API
        of the class. This must have a yield statement.
        """
        raise NotImplementedError()
    
    def __next__(self):
        """
        Return the next element in the generator.
        Args:
            numpy.array: next row of the generator
        """
        try:
            return next(self.generator)
        except StopIteration as e:
            self._iterator_end() # to the end of the iterator
            raise(e)
    
    def rewind(self):
        """ Rewind the generator. """
        self.generator = self._generator(**self.gen_kwargs)
    
    def _iterator_end(self):
        """ End of iterator logic. """
        pass


class TAStreamer(DataGenerator):
    """
    Data generator from .csv or .xlsx file(s).
    Args:
        fpath (str): Filepath to a csv or xlsx file.
    """
    @staticmethod
    def _generator(fpath, split = 0.8, \
                   mode = "train", spread = Spread, \
                   prep_isreturnbasis = Prep_IsReturnBasis):
        """ Generator function for single file """
        
        fext = os.path.splitext(fpath)[-1] # 取得副檔名
        # 不讀取索引欄(日期)；技術指標與報酬率需包含在檔案內
        if fext.lower() == ".csv":
            df = pd.read_csv(fpath, index_col = None, \
                             delimiter = myParams.Delimiter, \
                             skiprows = None, header = 0)
            # 將缺少技術指標數據的時間區間資料刪除
            # e.g.技術指標若包含SMA-60，則前59筆資料將被刪除
            df = df[Cols_1 + Cols_2].dropna(how = "any", axis = 0)
        elif fext.lower() == ".xlsx":
            df = pd.read_excel(fpath, index_col = None, \
                               skiprows = None, header = 0)
            # 將缺少技術指標數據的時間區間資料刪除
            # e.g.技術指標若包含SMA-60，則前59筆資料將被刪除
            df = df[Cols_1 + Cols_2].dropna(how = "any", axis = 0)
        else:
            Logger.error("Undefined file extension!")
        # 資料前處理
        # 注意：每個環境之資料正交化(normalization)均為分開進行
        # 只需要確保同一環境下訓練與測試之正交化一致即可
        if not prep_isreturnbasis:
            # Scale all data in the dataframe, including VOHLC
            min_max_scaler = prep.MinMaxScaler((-1, 1))
            np_scaled = min_max_scaler.fit_transform(df[Cols_1])
            df_scaled = pd.DataFrame(data = np_scaled, columns = Cols_1)
        else: # Use return-basis data
            # 使用 MinMaxScaler 對報酬率(和昨量比)做前處理
            min_max_scaler = prep.MinMaxScaler((-1, 1))
            np_scaled = min_max_scaler.fit_transform(df[Cols_2])
            df_scaled = pd.DataFrame(data = np_scaled, columns = Cols_2)
        # Bid price as demand index:
        # bid, ask, mid are not scaled
        
        df.reset_index(drop = True, inplace = True) # Reset index
        # df_scaled["bid"] = df['C'] # for index reset testing
        df_scaled["bid"] = (df['H'] + df['L'] + 2.0*df['C'])/4.0
        df_scaled["ask"] = df_scaled["bid"]*(1.0 + spread) # 台股
        # df_scaled["ask"] = df_scaled["bid"] + spread
        df_scaled["mid"] = (df_scaled["bid"] + df_scaled["ask"])/2.0
        
        split_len = int(split*df_scaled.shape[0])
        # 資料前段設為訓練集，後段為測試集，比例 = split
        if mode == "train":
            raw_data = df_scaled.iloc[:split_len,:]
        else: # mode == "test"
            raw_data = df_scaled.iloc[split_len:,:]
        for index, row in raw_data.iterrows():
            # Use yield statement for python generators
            # yield row.as_matrix() # depreciated
            yield row.values
    
    def _iterator_end(self):
        """
        Rewinds if end of data reached.
        """
        super().rewind()
    
    def rewind(self):
        """
        For this generator, we want to rewind only when the 
        end of the data is reached.
        """
        self._iterator_end()

class Env(object):
    """
    Abstract class for an environment. Simplified OpenAI API.
    """
    def __init__(self):
        """ Initialization """
        self.n_actions = None
        self.state_shape = None
    
    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple 
        (observation, reward, done, info).
        
        Args: action (numpy.array): action array
        
        Returns: tuple:
            - observation (numpy.array): 
                Agent's observation of the current environment.
            - reward (float) : 
                Amount of reward returned after previous action.
            - done (bool): 
                Whether the episode has ended, in which case further step() 
                calls will return undefined results.
            - info (str): Contains auxiliary diagnostic information 
                (helpful for debugging, and sometimes learning).
        """
        raise NotImplementedError()
    
    def reset(self):
        """
        Reset the state of the environment & returns an initial observation.
        Returns:
            numpy.array: The initial observation of the space. 
            Initial reward is assumed to be 0.
        """
        raise NotImplementedError()
    
    def render(self):
        """
        Render the environment.
        """
        raise NotImplementedError()


class Indicator_1(Env):
    """
    Class for a discrete (buy/hold/sell) spread trading environment.
    """
    # Defined key-values, privately used
    _actions = {
        'hold': np.array([1, 0, 0]),
        'buy': np.array([0, 1, 0]),
        'sell': np.array([0, 0, 1])
    }
    
    _positions = {
        'flat': np.array([1, 0, 0]),
        'long': np.array([0, 1, 0]),
        'short': np.array([0, 0, 1])
    }
    
    def reset(self):
        """
        Reset the trading environment to the initial observation.
        Returns:
            observation (numpy.array): observation of the state
        """
        self.trade_record = [] # 交易明細
        self.trade_record.append(["long_short", 
                                  "quantities", 
                                  "entry_price", 
                                  "exit_price", 
                                  "hold_period", 
                                  "profit_and_loss", 
                                  "total_fees", 
                                  "net_profit_and_loss"])
        self.asset_record = [] # 資產變動明細
        self.asset_record.append(["iteration", 
                                  "avg_price", 
                                  "cash_and_stock", 
                                  "cash", 
                                  "stock_value", 
                                  "action", 
                                  "position"])
        self._fees = 0. # 用來計算建倉至平倉之總交易和時間成本(負值)
        self._current_cash = self._initial_cash # Reset cash
        self._current_sec_value = 0. # 證券市值
        self._current_asset = self._current_cash + self._current_sec_value
        self._target_quantity = self._unit_quantity # 初期規劃之每筆交易量
        self._holdperiods = -1 # 持有天數；負值表示尚未起算
        self._time_fee_sum = 0.
        self._no_trades = 0 # 交易次數(建倉-平倉算一次)
        self._first_error_show = True # for debugging purpose
        self._terminated_info = ''
        self._iteration = 0 # start from the first day
        self._data_generator.rewind() # rewind to the first day
        self._position = self._positions["flat"]
        self._prices_history = []
        self._tick_buy = 0
        self._tick_sell = 0
        self.tick_mid = 0
        self._price = 0
        self._entry_price = 0
        self._exit_price = 0
        self.TP_render = False
        self.SL_render = False
        self.Buy_render = False
        self.Sell_render = False
        self._total_reward = 0
        self._total_pnl = 0 # pnl = profit and loss
        self.unrl_pnl = 0 # unrealized pnl
        self._closed_plot = False        
        self._first_render = True
        # 歷史觀察資料的寫入：
        for i in range(self._history_length):
            self._prices_history.append(next(self._data_generator))
        self.asset_record.append([self._iteration, 
                                  self.tick_mid, 
                                  self._current_asset, 
                                  self._current_cash, 
                                  self._current_sec_value, 
                                  0, 
                                  0])
        # 讀取最近一筆資料，寫入 tick prices
        # 買方出價/bid, 賣方出價/ask, 均報價/mid
        # 假設均為內盤成交，即成交賣價 = 買方出價，成交買價 = 賣方出價
        self._tick_sell, self._tick_buy, self.tick_mid = \
            self._prices_history[-1][-3:] # 最後三欄 = bid, ask, mid
        observation = self._get_observation() # 取得觀察資訊
        self.state_shape = observation.shape
        self._action = self._actions["hold"] # 初始：不採取動作
        return observation # initial state
    
    def __init__(self, data_generator, 
                 episode_length = 100, 
                 trading_tax = 0, 
                 trading_fee = 0, 
                 time_fee = 0, 
                 history_length = 2):
        """
        Initialisation function
        Args:
            data_generator (tgym.core.DataGenerator):
                A data generator object yielding a 1D array of bid-ask price.
            episode_length (int): 
                number of steps to play the game for
            trading_fee & trading_tax (float): 
                penalties for trading
            time_fee (float): 
                penalty for holding position
            history_length (int): 
                number of historical states to stack in the
                observation vector.
        """
        # 以下參數只需要初始化一次：
        assert history_length > 0
        self._data_generator = data_generator
        self._trading_tax = trading_tax # 新增，採比例扣除
        self._trading_fee = trading_fee # 改採比例扣除
        self._time_fee = time_fee # 改採比例扣除
        self._episode_length = episode_length
        self._history_length = history_length # 觀察天期，至少要有一筆(用於計算報酬)
        self._initial_cash = Initial_Cash
        self._unit_quantity = Unit_Quantity # 新增，單位交易量(股)
        self._max_lost = Max_Underwater_Ratio * Initial_Cash
        self._keep_cash_ratio = Keep_Cash_Ratio # 若為1，表示每次交易都 all-in
        self.n_actions = 3 # defined but not used
        self.reset() # 其餘類別參數以 reset() 初始化
    
    def step(self, action, aug_info = False):
        """
        Take an action (buy/sell/hold) and computes the immediate reward.
        Args:
            action (numpy.array): Action to be taken, one-hot encoded.
            aug_info (bool): 在目前版本新增，用來取得額外環境資訊
        Returns: tuple:
            - observation (numpy.array): 
                Agent's observation of the current environment.
            - reward (float) : 
                Amount of reward returned after previous action.
            - done (bool): 
                Whether the episode has ended, in which case further step() 
                calls will return undefined results.
            - info (dict): 
                Contains auxiliary diagnostic information 
                (helpful for debugging, and sometimes learning).
        """
        self._action = action
        self._iteration += 1
        done = False
        info = {}
        instant_pnl = 0
        reward = 0.
        lost = 0.
        time_fee_tuned = self._time_fee
        if self._holdperiods >= 0: # 持有證券中，須計支出利息
            self._holdperiods += 1
            # 將時間成本的計算改成比例，其中 _price 為進場價非市價
            # 由於持有證券時間成本高於持有現金，因此預期 
            # RL agent 只會在有行情時把握機會持有證券並短線操作
            if not Hold_Period_Upper is None: # 對時間成本做懲罰
                time_fee_tuned = \
                    Penaltize_Time_Fee(time_fee = self._time_fee, \
                                       t_hold = self._holdperiods)
            self._time_fee_sum += time_fee_tuned
            reward -= self._price * self._target_quantity * time_fee_tuned
            # 更新證券庫存市值，庫存量為建倉時的交易量
            # 在持有證券期間不會有現金流變化
            if all(self._position == self._positions["short"]):
                self._current_sec_value = self._target_quantity * \
                    (2. * self._price - self.tick_mid)
            else:
                self._current_sec_value = self._target_quantity * \
                    self.tick_mid
            # 總資產 = 證券市值 + 現金
            self._current_asset = self._current_sec_value + self._current_cash
        else: # 空手時須追蹤市價規劃交易量(若打算固定交易量則可以把這段註解掉)
            # 利用目前的 mid price 與資產，規劃交易量
            # keep_cash_ratio 若設太大容易 game over
            trade_cash = self._current_cash * (1.0 - self._keep_cash_ratio)
            if self.tick_mid > myParams.Tiny:
                self._target_quantity = int(trade_cash / self.tick_mid)
                if self._target_quantity < 1 and self._first_error_show:
                    # 交易現金過少或是價位太高，使成交不足一股！
                    Logger.error("Quantity < 1! Qty: %d" % self._target_quantity)
                    Logger.error("Check tick price: %.2f" % self.tick_mid)
                    Logger.error("Check trade cash: %.2f" % trade_cash)
                    self._first_error_show = False
                    self._target_quantity = 1 # 至少要交易一股
            else:
                if self._first_error_show:
                    Logger.error("Tick price abnormal: %s" % str(self.tick_mid))
                    self._first_error_show = False
                self._target_quantity = self._unit_quantity
        
        self.Buy_render = False
        self.Sell_render = False
        self.TP_render = False
        self.SL_render = False
        self._fees = 0. # not used
        
        # Agent's action | current state -> reward
        if all(self._action == self._actions["buy"]):
            if all(self._position == self._positions["flat"]): # 做多買進
                self._position = self._positions["long"] # 部位變更
                self._holdperiods = 0 # 起算持有天數
                # 取得買進價格與觀察價格 self._price
                self._entry_price = self._price = self._tick_buy
                # 將交易成本的計算改成比例
                reward -= self._entry_price * self._target_quantity * \
                    self._trading_fee
                self.Buy_render = True
            elif all(self._position == self._positions["short"]): # 融券回補
                # 取得平倉價格
                self._exit_price = self._tick_buy # 修正
                # 將交易成本的計算改成比例
                reward -= self._exit_price * self._target_quantity * \
                    (self._trading_fee + self._trading_tax)
                self._fees = self._entry_price * self._trading_fee + \
                    self._price * self._time_fee_sum + \
                    self._exit_price * (self._trading_fee + self._trading_tax)
                self._fees *= -self._target_quantity
                # 放空已實現損益(不計入交易成本)
                instant_pnl = (self._entry_price - self._exit_price) * \
                    self._target_quantity
                self.trade_record.append(["Short", 
                                          self._target_quantity, 
                                          round(self._entry_price, 4), 
                                          round(self._exit_price, 4), 
                                          self._holdperiods, 
                                          round(instant_pnl, 2), 
                                          round(self._fees, 2), 
                                          round((self._fees + instant_pnl), 2)])
                self._position = self._positions["flat"] # 平倉
                self._holdperiods = -1 # 重設
                self._time_fee_sum = 0. # 歸零
                self._no_trades += 1 # 交易次數+1
                
                # if (self._fees + instant_pnl > 0):
                if (instant_pnl > 0):
                    self.TP_render = True # take profit
                else:
                    self.SL_render = True # stop loss
        elif all(self._action == self._actions["sell"]):
            if all(self._position == self._positions["flat"]): # 融券放空
                self._position = self._positions["short"]
                self._holdperiods = 0 # 起算持有天數
                # 取得賣出價格與觀察價格 self._price
                self._entry_price = self._price = self._tick_sell
                # 將交易成本的計算改成比例
                reward -= self._entry_price * self._target_quantity * \
                    self._trading_fee
                self.Sell_render = True
            elif all(self._position == self._positions["long"]): # 多單平倉
                self._exit_price = self._tick_sell # 修正
                # 將交易成本的計算改成比例
                reward -= self._exit_price * self._target_quantity * \
                    (self._trading_fee + self._trading_tax)
                self._fees = self._entry_price * self._trading_fee + \
                    self._price * self._time_fee_sum + \
                    self._exit_price * (self._trading_fee + self._trading_tax)
                self._fees *= -self._target_quantity
                # 放空已實現損益(不計入交易成本)
                instant_pnl = (self._exit_price - self._entry_price) * \
                    self._target_quantity
                self.trade_record.append(["Long", 
                                          self._target_quantity, 
                                          round(self._entry_price, 4), 
                                          round(self._exit_price, 4), 
                                          self._holdperiods, 
                                          round(instant_pnl, 2), 
                                          round(self._fees, 2), 
                                          round((self._fees + instant_pnl), 2)])
                self._position = self._positions["flat"]
                self._holdperiods = -1 # 重設
                self._time_fee_sum = 0. # 歸零
                self._no_trades += 1 # 交易次數+1
                
                # if (self._fees + instant_pnl > 0):
                if (instant_pnl > 0):
                    self.TP_render = True
                else:
                    self.SL_render = True
        
        # 在沒有交易動作時，報酬數值仍會改變(時間成本)
        reward += instant_pnl
        self._total_pnl += instant_pnl # 不計入交易成本之盈虧
        self._total_reward += reward # 報酬，包含交易成本
        # 現金流變化對應資產變化：
        if self.Buy_render or self.Sell_render: # 建倉
            self._current_cash -= self._entry_price * \
                self._target_quantity * (1 + self._trading_fee)
            self._current_sec_value = self._target_quantity * self.tick_mid
            self._current_asset = self._current_sec_value + self._current_cash
        elif self.TP_render or self.SL_render: # 平倉
            # 取得現金 = 買進時的價金 + 價差獲利 - 平倉交易成本與持倉利息
            # 上述公式做多做空均適用
            self._current_cash += (instant_pnl + \
                                   self._entry_price * self._target_quantity)
            # 買進證券費必須從總交易成本中扣除
            # (注意 self._fees < 0)
            self._current_cash += (self._fees + self._entry_price * \
                                   self._trading_fee * self._target_quantity)
            self._current_sec_value = 0.
            self._current_asset = self._current_sec_value + self._current_cash
        # Debug:
        if self._current_cash < 0 and self._first_error_show:
            Logger.error("Unexpectedly debted! Current cash: %.2f" % \
                         self._current_cash)
            self._first_error_show = False
        elif self._current_asset < 0 and self._first_error_show:
            Logger.error("Unexpectedly bankrupted! Current asset: %.2f" % \
                         self._current_asset)
            self._first_error_show = False
        
        # 評估當前損失
        lost = min(0, self._current_asset - self._initial_cash)
        
        # 寫入記錄
        act_val = 0
        pstn_val = 0
        if all(self._action == self._actions['buy']):
            act_val = 1
        elif all(self._action == self._actions['sell']):
            act_val = -1
        if all(self._position == self._positions['long']):
            pstn_val = 1
        elif all(self._position == self._positions['short']):
            pstn_val = -1
        self.asset_record.append([self._iteration, 
                                  round(self.tick_mid, 4), 
                                  round(self._current_asset, 2), 
                                  round(self._current_cash, 2), 
                                  round(self._current_sec_value, 2), 
                                  act_val, 
                                  pstn_val])
        
        try:
            # 寫入歷史價格資訊
            self._prices_history.append(next(self._data_generator))
            # 買方出價/bid, 賣方出價/ask, 均報價/mid
            # 假設均為內盤成交，即成交賣價 = 買方出價，成交買價 = 賣方出價
            self._tick_sell, self._tick_buy, self.tick_mid = \
                self._prices_history[-1][-3:] # 最後三欄 = bid, ask, mid
        except StopIteration:
            done = True
            info["status"] = "No more data."
        
        if not self._first_error_show: # 在 logger 輸出過錯誤訊息
            done = True
            info["debug"] = "Error once caused!"
        
        # Game over logic
        iter_length = self._episode_length - self._history_length
        if self._iteration >= iter_length:
            done = True
            info["status"] = "Time out."
        if lost <= self._max_lost: # 超出可承受最大虧損
            done = True
            info["status"] = "Bankrupted ..."
        if self._closed_plot:
            # done is still set to False
            info["status"] = "Closed plot"
        
        if "status" in info:
            self._terminated_info = info["status"]
        
        # 取得額外資訊
        if aug_info: # output additional info (helpful to debugging)
            info["final_iter"] = self._iteration
            info["no_trades"] = self._no_trades
            info["total_pnl"] = round(self._total_pnl, 2)
            info["total_reward"] = round(self._total_reward, 2)
            info["total_lost"] = round(lost, 2)
        
        # 觀測取得下一個 state，並回傳
        observation = self._get_observation()
        return observation, reward, done, info
    
    def render(self, savefig = False, filename = "Indicator_1_Render"):
        """
        Matlplotlib rendering of each step.
        Args:
            savefig (bool): Whether to save the figure as an image or not.
            filename (str): Name of the image file.
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
        
        if self._first_render:
            self._f, (self._ax, self._at) = \
                plt.subplots(nrows = 2, ncols = 1, sharex = True, \
                             sharey = False, squeeze = True, \
                             gridspec_kw = {"height_ratios": [.6, .4]},)
            self._ax = [self._ax]
            self._at = [self._at]
            self._f.set_size_inches(12, 6)
            self._first_render = False
            # Event handling: current only close event is applied
            self._f.canvas.mpl_connect("close_event", self._handle_close)
        
        # ax -> 買賣進出交易圖
        ask, bid, mid = self._tick_buy, self._tick_sell,self.tick_mid
        self._ax[-1].plot([self._iteration, \
                           self._iteration + 1], [mid, mid], color = "white")
        self._ax[-1].set_ylabel("Tick price")
        self._ax[-1].grid(which = "major", axis = "both", linestyle = '-', \
                          color = "gray", linewidth = 0.5)
        
        ymin, ymax = self._ax[-1].get_ylim()
        yrange = ymax - ymin
        yshift = 0.05 * yrange
        xview = 90.5 # horizonal window view
        comment = ' '
        if self.Sell_render:
            self._ax[-1].scatter(self._iteration + 0.5, bid + yshift, \
                                 color = "lawngreen", marker = 'v')
            comment = "sell/short"
        elif self.Buy_render:
            self._ax[-1].scatter(self._iteration + 0.5, ask - yshift, \
                                 color = "orangered", marker = '^')
            comment = "buy/long"
        elif self.TP_render:
            self._ax[-1].scatter(self._iteration + 0.5, bid + yshift, \
                                 color = "gold", marker = '.')
            comment = "take profit"
        elif self.SL_render:
            self._ax[-1].scatter(self._iteration + 0.5, ask - yshift, \
                                 color = "maroon", marker = '.')
            comment = "stop loss"
        
        self.TP_render = False
        self.SL_render = False
        self.Buy_render = False
        self.Sell_render = False
        
        # at -> 帳戶水位變化圖
        # 繪製 self._current_asset/cash/sec_value 對時間的變化
        asset = self._current_asset
        cash = self._current_cash
        stock = self._current_sec_value
        val0 = self._initial_cash
        val1 = self._initial_cash + self._max_lost # lost是負值
        self._at[-1].plot([self._iteration, self._iteration + 1], \
                          [asset, asset], color = "white")
        self._at[-1].plot([self._iteration, self._iteration + 1], \
                          [cash, cash], color = "gold")
        self._at[-1].plot([self._iteration, self._iteration + 1], \
                          [stock, stock], color = "cyan")
        self._at[-1].plot([self._iteration, self._iteration + 1], \
                          [val0, val0], color = "green")
        self._at[-1].plot([self._iteration, self._iteration + 1], \
                          [val1, val1], color = "red")
        self._at[-1].set_ylabel("Values in NTD")
        at_label = ["Asset", "Cash", "Stock", "Initial asset", "Min. asset allowed"]
        self._at[-1].legend(labels = at_label, loc = "upper left", \
                            fancybox = True, framealpha = 0.5)
        self._at[-1].grid(which = "major", axis = "both", linestyle = '-', \
                          color = "gray", linewidth = 0.5)
        
        status = "running ..."
        if self._terminated_info != '':
            status = self._terminated_info
        plt.suptitle("Total Reward: " + '%.2f' % self._total_reward +
                     "  Total PnL: " + '%.2f' % self._total_pnl +
                     "  Unrealized Return: " + '%.2f' % (self.unrl_pnl*100)  + '% ' +
                     "\n  Position: " + ["flat", "long", "short"][list(self._position).index(1)] + 
                     "  Action: " + ["hold", "buy", "sell"][list(self._action).index(1)] + 
                     "  Action result: " + comment + 
                     "  Day: " + '%d' % self._iteration + 
                     "  Tick price: " + '%.2f' % self.tick_mid + 
                     "\n  Interaction status: " + status)
        self._f.tight_layout(pad = 0.4, w_pad = 0.5, h_pad = 1.0)
        plt.xticks(range(self._iteration)[::5])
        plt.xlim([max(0, self._iteration - xview), self._iteration + 0.5])
        plt.subplots_adjust(top = 0.85)
        plt.pause(0.001)
        if savefig:
            fdir1 = Trade_Path + "\\anime"
            fdir2 = fdir1 + '\\' + filename
            if not os.path.isdir(fdir1):
                os.mkdir(fdir1)
            if not os.path.isdir(fdir2):
                os.mkdir(fdir2)
            a = xview - 0.5
            # Save images as specified steps or last step reached!
            # 關閉視窗時存取最後一步畫面
            # self._iteration 不應大於 iter_length，若大於則每步都輸出圖檔
            if (self._iteration >= a and self._iteration % 10 == 0) or \
                (self._terminated_info != ''):
                fpath = fdir2 + '\\' + filename +  '_' + \
                    str(self._iteration) + ".png" # for making .gif animation
                plt.savefig(fpath)
        # return None
    
    def _handle_close(self, evt):
        """ 輔助函式，由render()呼叫 """
        # Callback 函式必須額外提供引數
        # 在此引數 evt 為輸入之 event object.
        self._closed_plot = True
    
    def return_calc(self, render_show = False):
        """ 對帳單輸出 """
        trade_details = {}
        # 進場(建倉) -> 做多或做空
        if self.Sell_render:
            trade_details = {"Trade": "SELL (SHORT)", 
                             "Price": round(self._entry_price, 4), 
                             "Time": self._iteration}
        elif self.Buy_render:
            trade_details = {"Trade": "BUY (LONG)", 
                             "Price": round(self._entry_price, 4), 
                             "Time": self._iteration}
        # 出場(平倉) -> TP/SL表示(未計入交易成本之)盈/虧
        if self.TP_render:
            trade_details = {"Trade": "TAKE PROFIT (TP)", 
                             "Price": round(self._exit_price, 4), 
                             "Time": self._iteration}
        elif self.SL_render:
            trade_details = {"Trade": "STOP LOSS (SL)", 
                             "Price": round(self._exit_price, 4), 
                             "Time": self._iteration}
        
        if(not render_show):
            self.TP_render = False
            self.SL_render = False
            self.Buy_render = False
            self.Sell_render = False
        return trade_details
    
    def return_prices_history(self):
        """ (除錯與單元測試用) 回傳歷史資料 """
        return self._prices_history
    
    def export_trade_records(self, filename = "Indicator_1_Trade_Record"):
        """ 將交易記錄輸出至 .csv 檔 """
        fpath = Trade_Path + '\\' + filename + "_trade_rec.csv"
        data = []
        cols = []
        if len(self.trade_record) > 1:
            data = self.trade_record[1:]
            cols = self.trade_record[0]
        df = pd.DataFrame(data = data, columns = cols)
        df.to_csv(fpath, sep = ',', header = True, index = False)
        return None
    
    def export_asset_records(self, filename = "Indicator_1_Asset_Record"):
        """ 將資產水位記錄輸出至 .csv 檔 """
        fpath = Trade_Path + '\\' + filename + "_asset_rec.csv"
        data = []
        cols = []
        if len(self.asset_record) > 1:
            data = self.asset_record[1:]
            cols = self.asset_record[0]
        df = pd.DataFrame(data = data, columns = cols)
        df.to_csv(fpath, sep = ',', header = True, index = False)
        return None
    
    def get_action_size(self):
        """ 取得已定義的動作數量 """
        return len(self._actions)
    
    def _get_observation(self):
        """
        Concatenate all necessary elements to create the observation.
        Returns:
            numpy.array: observation array.
        """
        if all(self._position == self._positions["flat"]):
            self.unrl_pnl = 0
        elif all(self._position == self._positions["long"]):
            self.unrl_pnl = (self._prices_history[-1][-1] - self._price) \
                / self._prices_history[-1][-1]
        elif all(self._position == self._positions["short"]):
            self.unrl_pnl = (self._price - self._prices_history[-1][-1]) \
                / self._prices_history[-1][-1]
        # 回傳 state 時，不需要 bid, ask, mid prices
        # 依據history_length設定來決定回傳多少根K棒資料
        # 例如history_length = 3則回傳今日+昨日K棒資訊
        his_k = []
        if self._history_length > 2:
            for i in range(self._history_length - 2):
                kk = -2 - i
                if not IsVolUsed: # O, H, L, C
                    his_k.append(self._prices_history[kk][0:4])
                else: # V, O, H, L, C
                    his_k.append(self._prices_history[kk][0:5])
        
        return np.concatenate(
            [self._prices_history[-1][:-3]] + 
            his_k + 
            [
                np.array([self.unrl_pnl]),
                np.array(self._position)
            ]
        )
    
    @staticmethod
    def random_action_fun(prob_hold = 0.8):
        """
        The default random action for exploration.
        We hold 80% of the time and buy or sell 10% of the time each.
        
        Returns:
            numpy.array: array with a 1 on the action index, 0 elsewhere.
        """
        # 對環境毫無觀察能力下，以隨機方式進行動作
        x = prob_hold # default 0.8
        y = z = (1.0 - prob_hold)/2.0 # default 0.1 each
        return np.random.multinomial(1, [x, y, z])

# Done!
