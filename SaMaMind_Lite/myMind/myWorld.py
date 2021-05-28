# -*- coding: utf-8 -*-

import os
import datetime as dt
import numpy as np
import pandas as pd
import re
from random import shuffle

from myUtilities import myParams
from myUtilities import myLog
from myStockScraper.myPostProcessing import TradePeriod
from myMind import myAgent
from myMind.myAgent import DQNAgent, DDQNAgent, DDDQNAgent
from myMind import myEnv
from myMind.myEnv import TAStreamer, Indicator_1
from myMind import myWorldSettings # 已定義之參數組
from myMind import myDNNVisualizer as vis # 模型視覺化
from myMind import myAna # 模型效能分析

"""
參考 Saeed Rahman 等人的研究和程式碼
來源：https://github.com/saeed349/Deep-Reinforcement-Learning-in-Trading
myWorld.py 由 Main.py 改寫而成
其他可參考：
https://github.com/ThibautTheate/An-Application-of-Deep-Reinforcement-Learning-to-Algorithmic-Trading

myWorld.py 新增功能：
(1) 可針對複數組資料做訓練與測試
(2) 可選擇 RL agent
(3) AI 參數套組：針對以下需求作不同定義設定 (設置於 myWorldSettings.py)
"""

# Global variables and constants
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

Test_Path = RefPath + '\\' + myParams.PostP_Foldername # for prediction
Train_Path = Test_Path # training & validation; may set to be different path

# 交易成本不考慮電子單六折&最低手續費，也不考慮當沖證交稅減免等特殊規則
# 因此模擬交易所使用的交易成本會略高於實際下單的成本
Default_Trading_Tax = 0.003 # 證交稅(平倉時徵收)
Default_Trading_Fee = 0.001425 # 券商買&賣手續費
# Time fee 只考慮定存機會成本，不考慮融資券利率
Default_Time_Fee = 0.01/365 # 無風險年利率 = 1% 
Default_Train_Test_Split = 0.8 # 80% for training, 20% testing
Default_Episode_Length = 1000
Default_Episodes = 100
Default_History_Length = 2 # Indicator_1.reset() 中預載入資料觀察天期數

Trading_Tax = Default_Trading_Tax
Trading_Fee = Default_Trading_Fee
Time_Fee = Default_Time_Fee
Train_Test_Split = Default_Train_Test_Split
Episode_Length = Default_Episode_Length
Episodes = Default_Episodes
History_Length = Default_History_Length

# 注意：在函式Training與Testing的引數預設值所對應的是全域變數的初值
# 無論全域變數如Train_Test_Split之後如何修改，train_test_split的預設值
# 仍為其最初值，因此在呼叫Training與Testing時最好所有引數都再傳遞一次！

# --- --- #

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def WorldReset():
    """ 還原初始設定 """
    global Trading_Tax, Trading_Fee, Time_Fee, Train_Test_Split
    global Episode_Length, Episodes, History_Length
    Trading_Tax = Default_Trading_Tax
    Trading_Fee = Default_Trading_Fee
    Time_Fee = Default_Time_Fee
    Train_Test_Split = Default_Train_Test_Split
    Episode_Length = Default_Episode_Length
    Episodes = Default_Episodes
    History_Length = Default_History_Length
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def WorldSetUp(**kwargs):
    """
    設定RL模擬預設參數。已定義關鍵字：
    trading_tax, (float, [0, ) )
    trading_fee, (float, [0, ) )
    time_fee, (float, [0, ) )
    train_test_split, (float, [0, 1] )
    episode_length, (int [1, ) )
    episodes, (int [1, ) )
    history_length, (int [1, ) )
    """
    global Trading_Tax, Trading_Fee, Time_Fee, Train_Test_Split
    global Episode_Length, Episodes, History_Length
    WorldReset()
    
    if "trading_tax" in kwargs:
        if type(kwargs["trading_tax"]) is float and \
            kwargs["trading_tax"] >= 0:
            Trading_Tax = kwargs["trading_tax"]
        else:
            Logger.warning("Invalid trading_tax (required: > 0.0).")
    if "trading_fee" in kwargs:
        if type(kwargs["trading_fee"]) is float and \
            kwargs["trading_fee"] >= 0:
            Trading_Fee = kwargs["trading_fee"]
        else:
            Logger.warning("Invalid trading_fee (required: > 0.0).")
    if "time_fee" in kwargs:
        if type(kwargs["time_fee"]) is float and kwargs["time_fee"] >= 0:
            Time_Fee = kwargs["time_fee"]
        else:
            Logger.warning("Invalid time_fee (required: > 0.0).")
    if "train_test_split" in kwargs:
        if type(kwargs["train_test_split"]) is float \
            and kwargs["train_test_split"] >= 0.0 and \
            kwargs["train_test_split"] <= 1.0:
            Train_Test_Split = kwargs["train_test_split"]
        else:
            Logger.warning("Invalid train_test_split!")
            Logger.warning("(Required: >= 0.0 and <= 1.0)")
    if "episode_length" in kwargs:
        if int(kwargs["episode_length"]) >= 1:
            Episode_Length = int(kwargs["episode_length"])
        else:
            Logger.warning("Invalid episode_length (required: >= 1).")
    if "episodes" in kwargs:
        if int(kwargs["episodes"]) >= 1:
            Episodes = int(kwargs["episodes"])
        else:
            Logger.warning("Invalid episodes (required: >= 1).")
    if "history_length" in kwargs:
        if int(kwargs["history_length"]) >= 1:
            History_Length = int(kwargs["history_length"])
        else:
            Logger.warning("Invalid history_length (required: >= 1).")
    # Examine the variables:
    Logger.info("trading_tax: %s" % str(Trading_Tax))
    Logger.info("trading_fee: %s" % str(Trading_Fee))
    Logger.info("time_fee: %s" % str(Time_Fee))
    Logger.info("train_test_split: %s" % str(Train_Test_Split))
    Logger.info("episode_length: %s" % str(Episode_Length))
    Logger.info("episodes: %s" % str(Episodes))
    Logger.info("history_length: %s" % str(History_Length))
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def ChangeTheWorld_V2(**kwargs):
    """
    給定待修改參數內容進行更新 (由myApp.py呼叫)
    以下更新myEnv/myAgent/myWorld的全域變數之後再將完整的參數組回傳
    """
    pset = {}
    if kwargs is None or kwargs == {}:
        Logger.info("Nonetype or empty kwarg. found!")
        Logger.info("Skip updating parameter set.")
    else:
        # 由於SetUp函式皆會先呼叫Reset，因此最好的做法是
        # **kwargs中包含所有全域變數的鍵值對
        # 可透過ChangeTheWorld(psetname, True)回傳值取得
        Logger.debug("Call WorldSetUp method ...")
        WorldSetUp(**kwargs)
        Logger.debug("Call myEnv.EnvSetUp method ...")
        myEnv.EnvSetUp(**kwargs)
        Logger.debug("Call myAgent.AgentSetUp method ...")
        myAgent.AgentSetUp(**kwargs)
    # 將所有全域變數傳回，並建立字典型態資料
    # myWorld 模組：
    pset["trading_tax"] = Trading_Tax
    pset["trading_fee"] = Trading_Fee
    pset["time_fee"] = Time_Fee
    pset["train_test_split"] = Train_Test_Split
    pset["episode_length"] = Episode_Length
    pset["episodes"] = Episodes
    pset["history_length"] = History_Length
    # myEnv 模組：
    pset["spread"] = myEnv.Spread
    pset["prep_isreturnbasis"] = myEnv.Prep_IsReturnBasis
    pset["initial_cash"] = myEnv.Initial_Cash
    pset["unit_quantity"] = myEnv.Unit_Quantity
    pset["max_underwater_ratio"] = myEnv.Max_Underwater_Ratio
    pset["keep_cash_ratio"] = myEnv.Keep_Cash_Ratio
    pset["hold_period_upper"] = myEnv.Hold_Period_Upper
    # myAgent 模組：
    pset["neurons_per_layer"] = myAgent.Neurons_Per_Layer
    pset["hiddenlayer_shape"] = myAgent.HiddenLayer_NParams_Ratio
    pset["chosen_agent"] = myAgent.Chosen_Agent
    pset["epsilon_min"] = myAgent.Epsilon_Min
    pset["gamma"] = myAgent.Gamma
    pset["activation"] = myAgent.Activation
    pset["l2_strength"] = myAgent.L2_Strength
    pset["learning_rate"] = myAgent.Learning_Rate
    pset["train_interval"] = myAgent.Train_Interval
    pset["update_target_freq"] = myAgent.Update_Target_Freq
    pset["memory_size"] = myAgent.Memory_Size
    pset["batch_size"] = myAgent.Batch_Size
    pset["validation_split"] = myAgent.Validation_Split
    return pset

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def ChangeTheWorld(paramsetname = None, isreturndict = False):
    """
    給定參數組的名稱，代入對應的全域變數
    參數組定義於 myWorldSettings.py
    """
    psetname = "default_test" # 預設參數組
    if paramsetname is None:
        Logger.info("Caution: param set name is not specified. Use default.")
        Logger.info("Param set is %s (for RL model testing)." % psetname)
    else:
        psetname = paramsetname
        Logger.info("Param set for changing the world: %s " % psetname)
    if psetname in myWorldSettings.RL_Set_Dict:
        pset = myWorldSettings.RL_Set_Dict[psetname]
    else:
        Logger.warning("Invalid arg.: %s" % psetname)
        Logger.warning("The world is not changed!")
        return None
    # Change global variables:
    Logger.debug("Call WorldSetUp method ...")
    WorldSetUp(trading_tax = pset.trading_tax, 
               trading_fee = pset.trading_fee, 
               time_fee = pset.time_fee, 
               train_test_split = pset.train_test_split, 
               episode_length = pset.episode_length, 
               episodes = pset.episodes, 
               history_length = pset.history_length)
    Logger.debug("Call myEnv.EnvSetUp method ...")
    myEnv.EnvSetUp(spread = pset.spread, 
                   prep_isreturnbasis = pset.prep_isreturnbasis, 
                   initial_cash = pset.initial_cash, 
                   unit_quantity = pset.unit_quantity, 
                   max_underwater_ratio = pset.max_underwater_ratio, 
                   keep_cash_ratio = pset.keep_cash_ratio, 
                   hold_period_upper = pset.hold_period_upper)
    Logger.debug("Call myAgent.AgentSetUp method ...")
    myAgent.AgentSetUp(neurons_per_layer = pset.neurons_per_layer, 
                       hiddenlayer_shape = pset.hiddenlayer_shape, 
                       chosen_agent = pset.chosen_agent, 
                       epsilon_min = pset.epsilon_min, 
                       gamma = pset.gamma, 
                       activation = pset.activation, 
                       l2_strength = pset.l2_strength, 
                       learning_rate = pset.learning_rate, 
                       train_interval = pset.train_interval, 
                       update_target_freq = pset.update_target_freq, 
                       memory_size = pset.memory_size, 
                       batch_size = pset.batch_size, 
                       validation_split = pset.validation_split)
    Logger.debug("Now return the param set named %s" % psetname)
    if isreturndict:
        # named_tuple -> ordered_dict -> dict/tuple -> str -> dict
        Logger.warning("Caution: complicated datatype conversion for pset!")
        return eval(str(dict(pset._asdict())))
    else: # return named_tuple
        return pset

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
# @myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def GetFpathList(folder = "train"):
    """
    取得訓練集或測試集檔案路徑清單
    folder = "train": load all .xlsx files in Train_Path
    folder = "test": load all .xlsx files in Test_Path
    """
    fplist = []
    ope_path = Test_Path # historical data
    fext = ".xlsx"
    if folder == "train":
        ope_path = Train_Path
        # fext = ".csv"
    if not os.path.isdir(ope_path):
        Logger.error("Target path %s does not exist!" % ope_path)
        return fplist # empty list
    flist_raw = os.listdir(ope_path)
    for ff in flist_raw:
        if os.path.splitext(ff)[-1].lower() == fext.lower():
            fpath = ope_path + '\\' + ff
            fplist.append(fpath)
    return fplist

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def Training(filename = None, 
             agent_name = myAgent.Chosen_Agent, 
             episodes = Episodes, 
             episode_length = Episode_Length, 
             train_test_split = Train_Test_Split, 
             history_length = History_Length, 
             time_fee = Time_Fee, 
             trading_fee = Trading_Fee, 
             trading_tax = Trading_Tax, 
             memory_size = myAgent.Memory_Size, 
             batch_size = myAgent.Batch_Size, 
             symbol = myAgent.Symbol, 
             save_results = False, 
             israndompick = False):
    """ 訓練並驗證 RL agent """
    Logger.info("Start training ...")
    train_test = "train"
    # Rawdata file gathering:
    if filename is None:
        fplist = GetFpathList(train_test) # file list
        Logger.info("Search files in train folder.")
        Logger.info("First file taken: %s" % fplist[0])
    elif type(filename) is int:
        fplist = GetFpathList(train_test) # file list
        if israndompick:
            Logger.debug("File path list will be shuffled!")
            shuffle(fplist) # 將檔案路徑清單洗牌
        if filename == 1:
            fplist = [fplist[0]]
        elif filename <= len(fplist):
            fplist = fplist[:filename]
        else:
            filename = len(fplist)
        Logger.info("Take first %d files in train folder." % filename)
        Logger.info("First file taken: %s" % fplist[0])
    else:
        fplist = [filename] # single file
        Logger.info("Single file training. File taken: %s" % fplist[0])
    
    # File preprocessing:
    genlist = []
    eplenlist = []
    for ff in fplist:
        # 產生訓練集資料
        generator = TAStreamer(fpath = ff, 
                               mode = train_test, 
                               split = train_test_split, 
                               spread = myEnv.Spread, 
                               prep_isreturnbasis = myEnv.Prep_IsReturnBasis)
        genlist.append(generator)
        
        eplen = history_length # data used during env.reset()
        done = False
        info = {}
        env = Indicator_1(data_generator = generator, 
                          episode_length = 10000, 
                          history_length = history_length)
        _ = env.reset()
        while not done:
            _, _, done, info = env.step(env.random_action_fun(1.0), True)
            eplen += 1
            if done:
                Logger.debug("Show env.step returned info: ")
                for k, v in info.items():
                    Logger.debug("{k} = {v}".format(k = k, v = v))
        eplenlist.append(eplen) # 取得每個檔案前處理後之資料長度
    eplen_min = min(eplenlist)
    Logger.debug("Found min. episode length = %d" % eplen_min)
    
    # episode length validation:
    if eplen_min < 100:
        Logger.warning("Narrow range (< 100 days) of training data found!")
        Logger.warning("Recommended: more than 200 days.")
        Logger.warning("Applicable (min. episode length): %d" % eplen_min)
        Logger.warning("Exit function ...")
        return None
    
    if episode_length > eplen_min:
        Logger.warning("Assigned episode length will be replaced!")
        Logger.warning("Assigned by user: %d" % episode_length)
        Logger.warning("Applicable (min. episode length): %d" % eplen_min)
        episode_length = eplen_min
    
    # Validate memory size:
    if (episode_length * len(genlist)) < memory_size:
        Logger.warning("Training data points is less than memory size!")
        dps = (episode_length - history_length - 1) * len(genlist)
        Logger.warning("Max allowed data points: %d" % dps)
        Logger.warning("Memory size assigned by user: %d" % memory_size)
        Logger.warning("Set memory_size to max allowed value and ")
        Logger.warning("batch_size = 1/8 of memory_size!")
        memory_size = dps
        batch_size = int(memory_size / 8)
    
    # 初始環境狀態設定
    environment = Indicator_1(data_generator = genlist[0], 
                              trading_fee = trading_fee, 
                              trading_tax = trading_tax, 
                              time_fee = time_fee, 
                              episode_length = episode_length, 
                              history_length = history_length)
    action_size = environment.get_action_size()
    state = environment.reset()
    state_size = len(state)
    
    # Choose and initialize RL agent for training:
    if agent_name == "DQN":
        # Deep Q-learning neural network
        # The basic version of deep RL agent
        agent = DQNAgent(state_size = state_size,
                         action_size = action_size,
                         memory_size = memory_size,
                         episodes = episodes,
                         episode_length = episode_length, 
                         no_generators = len(genlist), 
                         batch_size = batch_size,
                         train_test = train_test,
                         symbol = symbol, 
                         train_interval = myAgent.Train_Interval, 
                         update_target_freq = myAgent.Update_Target_Freq, 
                         gamma = myAgent.Gamma, 
                         learning_rate = myAgent.Learning_Rate, 
                         epsilon_min = myAgent.Epsilon_Min, 
                         validation_split = myAgent.Validation_Split)
    elif agent_name == "DDQN":
        # Double deep Q-learning neural network
        # Better than DQN by reducing Q-value overestimation
        agent = DDQNAgent(state_size = state_size, 
                          action_size = action_size, 
                          memory_size = memory_size, 
                          episodes = episodes, 
                          episode_length = episode_length, 
                          no_generators = len(genlist), 
                          batch_size = batch_size, 
                          train_test = train_test, 
                          symbol = symbol, 
                          train_interval = myAgent.Train_Interval, 
                          update_target_freq = myAgent.Update_Target_Freq, 
                          gamma = myAgent.Gamma, 
                          learning_rate = myAgent.Learning_Rate, 
                          epsilon_min = myAgent.Epsilon_Min, 
                          validation_split = myAgent.Validation_Split)
    else: # Default = DDDQN
        # Dueling double deep Q-learning neural network
        # Better than DDQN by introducing stream splitting (Q = V + A),
        # which enhancing the training efficiency and tesing performance!
        if agent_name != "DDDQN":
            Logger.info("Use default agent DDDQN!")
        agent = DDDQNAgent(state_size = state_size, 
                           action_size = action_size, 
                           memory_size = memory_size, 
                           episodes = episodes, 
                           episode_length = episode_length, 
                           no_generators = len(genlist), 
                           batch_size = batch_size, 
                           train_test = train_test, 
                           symbol = symbol, 
                           train_interval = myAgent.Train_Interval, 
                           update_target_freq = myAgent.Update_Target_Freq, 
                           gamma = myAgent.Gamma, 
                           learning_rate = myAgent.Learning_Rate, 
                           epsilon_min = myAgent.Epsilon_Min, 
                           validation_split = myAgent.Validation_Split)
    
    loss_list = []
    val_loss_list = []
    reward_list = []
    epsilon_list = []
    metrics_df = None
    try:
        # Warming up the agent with random action (full exploration)
        # 配置 memory，但 agent.observe() 還不會開始訓練模型
        Logger.debug("mem_size: %d, episode_len: %d, history_len: %d" % \
                     (memory_size, episode_length, history_length))
        counts = 0
        for i in range(len(genlist)):
            done = False
            info = {}
            environment = Indicator_1(data_generator = genlist[i], 
                                      trading_fee = trading_fee, 
                                      trading_tax = trading_tax, 
                                      time_fee = time_fee, 
                                      episode_length = episode_length, 
                                      history_length = history_length)
            state = environment.reset()
            for _ in range(episode_length):
                if counts >= memory_size:
                    Logger.debug("Memory refilled!")
                    break
                else:
                    counts += 1    
                if done:
                    Logger.debug("Show env.step returned info: ")
                    for key, value in info.items():
                        Logger.debug("{k} = {v}".format(k = key, v = value))
                    break
                action = agent.act(state)
                next_state, reward, done, info = environment.step(action)
                agent.observe(state, action, reward, next_state, \
                              done, warming_up = True)
            Logger.debug("gen_id: %d counts: %d" % (i, counts))
        if counts < memory_size:
            Logger.warning("Incomplete warming up!")
            Logger.warning("Target mem_size: %d, filled: %d" % \
                           (memory_size, counts))
        
        # Training:
        Logger.debug("# of data generators: %d" % len(genlist))
        Logger.info(" * * * * * * * * * * * * ")
        # Batch file mode of Saeed's World function (the training part)
        for ep in range(episodes):
            rew = 0
            loss_list_temp = []
            val_loss_list_temp = []
            for gen in genlist:
                info = {}
                done = False
                # Re-initialize and change the data generator
                environment = Indicator_1(data_generator = gen, 
                                          trading_fee = trading_fee, 
                                          trading_tax = trading_tax, 
                                          time_fee = time_fee, 
                                          episode_length = episode_length, 
                                          history_length = history_length)
                state = environment.reset()
                for _ in range(episode_length):
                    if done: # Episode finishes after previous iteration!
                        Logger.info("Show env.step returned info: ")
                        for key, value in info.items():
                            Logger.info("{k} = {v}".format(k = key, v = value))
                        break # to next data generator
                    action = agent.act(state)
                    next_state, reward, done, info = \
                        environment.step(action, True)
                    # Keras的fit方法會回傳 history loss，包含每個 epoch 的 loss
                    # 目前在 agent.observe() 中，參數 epochs = 1，故只會有一個值
                    # Below is Saeed's comment ...
                    # loss would be none if the episode length is not % by 10
                    loss = agent.observe(state, action, reward, next_state,
                                         done)
                    state = next_state
                    rew += reward
                    # Logger.debug("Current step: %d, loss type: %s" % \
                    #              (_, str(type(loss))))
                    if(loss):
                        loss_list_temp.append\
                            (round(loss.history["loss"][0], 4))
                        val_loss_list_temp.append\
                            (round(loss.history["val_loss"][0], 4))
                        Logger.debug("Ep: %s rew: %s eps: %s loss: %s val-loss: %s" % \
                                     (str(ep), str(round(rew, 2)), \
                                     str(round(agent.epsilon, 2)), \
                                     str(round(loss.history["loss"][0], 4)), \
                                     str(round(loss.history["val_loss"][0], 4))))
            Logger.debug("Check # of losses: %s c.f. # of generators: %s" % \
                         (str(len(loss_list_temp)), str(len(genlist))))
            Logger.debug("mean loss: %s, mean val_loss: %s" % \
                         (str(round(np.mean(loss_list_temp), 4)), \
                         str(round(np.mean(val_loss_list_temp), 4))))
            loss_list.append(np.mean(loss_list_temp))
            val_loss_list.append(np.mean(val_loss_list_temp))
            reward_list.append(round(rew, 2))
            epsilon_list.append(round(agent.epsilon, 4))
            Logger.info(" * * * * * * * * * * * * ")
        Logger.info("Training completed! Save RL agent model ...")
        agent.save_model()
    except KeyboardInterrupt: # interrupted by user
        Logger.error("Keyboard interrupt ocurred! Task terminated ...")
        Logger.error("Agent model will not be saved.")
    except Exception as e:
        Logger.critical("Unexpected exceptions occurred!")
        Logger.critical("{}".format(e))
    
    # Restore the current metric results
    metrics_df = pd.DataFrame({"loss": loss_list, 
                               "val_loss": val_loss_list, 
                               "reward": reward_list, 
                               "epsilon":epsilon_list})
    if save_results:
        metrics_df.to_csv(RefPath + '\\' + symbol + "_training.csv")
    Logger.info("All task done! Ready to exit training function ...")
    print("Training complete!")
    print("All task done! Ready to exit training function ...")
    return metrics_df

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def Testing(filename = None, 
            search_folder = "train", 
            agent_name = myAgent.Chosen_Agent, 
            train_test_split = Train_Test_Split, 
            history_length = History_Length, 
            time_fee = Time_Fee, 
            trading_fee = Trading_Fee, 
            trading_tax = Trading_Tax, 
            symbol = myAgent.Symbol, 
            save_results = False, 
            render_save = False, 
            render_show = False, 
            israndompick = False):
    """ 測試 RL agent 之表現，計算完成後可再確認其交易績效 """
    Logger.info("Start testing ...")
    # 檔案量可能爆炸多，先輸出注意訊息
    if render_show:
        print("Caution: So many graphs may be showed up!")
        Logger.debug("Caution: So many graphs may be showed up!")
    if save_results:
        print("Caution: So many files may be generated and exported, ")
        print("including .csv, .png, .pkl, ...")
        Logger.debug("Caution: So many files may be generated and exported, ")
        Logger.debug("including .csv, .png, .pkl, ...")
    if search_folder in ["train", "test"]:
        Logger.info("Search folder for %s set data. " % search_folder)
    else:
        search_folder = "train"
        Logger.warning("Invalid search folder arg.: " % search_folder)
        Logger.warning("Search folder for training set instead!")
    train_test = "test" # may test on either training or testing data set
    # Rawdata file gathering:
    if filename is None:
        fplist = GetFpathList(search_folder) # file list
        Logger.info("First file taken: %s" % fplist[0])
    elif type(filename) is int:
        fplist = GetFpathList(search_folder) # file list
        if israndompick:
            Logger.debug("File path list will be shuffled!")
            shuffle(fplist) # 將檔案路徑清單洗牌
        if filename == 1:
            fplist = [fplist[0]]
        elif filename <= len(fplist):
            fplist = fplist[:filename]
        else:
            filename = len(fplist)
        Logger.info("Take first %d files in folder." % filename)
        Logger.info("First file taken: %s" % fplist[0])
    else:
        fplist = [filename] # single file
        Logger.info("Single file testing. File taken: %s" % fplist[0])
    
    # File preprocessing:
    genlist = []
    eplenlist = []
    for i in range(len(fplist)):
        # 產生測試集資料
        generator = TAStreamer(fpath = fplist[i], 
                               mode = train_test, 
                               split = train_test_split, 
                               spread = myEnv.Spread, 
                               prep_isreturnbasis = myEnv.Prep_IsReturnBasis)
        genlist.append(generator)
        
        eplen = history_length # data used during env.reset()
        done = False
        info = {}
        env = Indicator_1(data_generator = generator, 
                          episode_length = 10000, 
                          history_length = history_length)
        _ = env.reset()
        while not done:
            _, _, done, info = env.step(env.random_action_fun(1.0), True)
            eplen += 1
            if done:
                Logger.debug("Show env.step returned info: ")
                for k, v in info.items():
                    Logger.debug("{k} = {v}".format(k = k, v = v))
        eplenlist.append(eplen) # 取得每個檔案前處理後之資料長度
    eplen_min = min(eplenlist)
    Logger.debug("Min. episode length: %d" % eplen_min)
    
    if eplen_min <= history_length:
        Logger.error("Some testing data is empty! Exit function ...")
        return None
    
    # 初始環境狀態設定
    environment = Indicator_1(data_generator = genlist[0], 
                              trading_fee = trading_fee, 
                              trading_tax = trading_tax, 
                              time_fee = time_fee, 
                              episode_length = eplenlist[0], 
                              history_length = history_length)
    action_size = environment.get_action_size()
    state = environment.reset()
    state_size = len(state)
    
    # Choose and initialize RL agent for testing:
    # episodes & episode_length are assigned but their values don't matter
    if agent_name == "DQN":
        # Deep Q-learning neural network
        # The basic version of deep RL agent
        agent = DQNAgent(state_size = state_size,
                         action_size = action_size,
                         episodes = 10,
                         episode_length = 100, 
                         no_generators = len(genlist), 
                         train_test = train_test, 
                         symbol = symbol)
    elif agent_name == "DDQN":
        # Double deep Q-learning neural network
        # Better than DQN by reducing Q-value overestimation
        agent = DDQNAgent(state_size = state_size, 
                          action_size = action_size, 
                          episodes = 10, 
                          episode_length = 100, 
                          no_generators = len(genlist), 
                          train_test = train_test, 
                          symbol = symbol)
    else: # Default = DDDQN
        # Dueling double deep Q-learning neural network
        # Better than DDQN by introducing stream splitting (Q = V + A),
        # which enhancing the training efficiency and tesing performance!
        if agent_name != "DDDQN":
            Logger.info("Use default agent DDDQN!")
        agent = DDDQNAgent(state_size = state_size, 
                           action_size = action_size, 
                           episodes = 10, 
                           episode_length = 100, 
                           no_generators = len(genlist), 
                           train_test = train_test, 
                           symbol = symbol)
    # Load the trained RL model
    agent.load_model()
    try:
        # Testing:
        Logger.debug("# of data generators: %d" % len(genlist))
        Logger.info(" * * * * * * * * * * * * ")
        # Batch file mode of Saeed's World function (the testing part)
        for ii in range(len(genlist)):
            # Set for the trade game
            done = False
            q_values_list = []
            state_list = []
            action_list = []
            reward_list = []
            trade_list = []
            # Extract filename in fplist via regexp:
            # fn = re.findall(r'[^\\]+(?=\.csv$)',fplist[ii])[-1]
            fn = re.findall(r'[^\\]+(?=\.)', fplist[ii])[-1]
            
            # Re-initialize and change the data generator
            environment = Indicator_1(data_generator = genlist[ii], 
                                      trading_fee = trading_fee, 
                                      trading_tax = trading_tax, 
                                      time_fee = time_fee, 
                                      episode_length = eplenlist[ii], 
                                      history_length = history_length)
            state = environment.reset()
            
            while not done:
                action, q_values = agent.act(state, test = True)
                state, reward, done, info = environment.step(action, True)
                if "status" in info and info["status"] == "Closed plot":
                    done = True
                else:
                    reward_list.append(reward)
                    calc_returns = environment.return_calc(render_show)
                    # 運算結果即時輸出與顯示
                    if calc_returns:
                        trade_list.append(calc_returns)
                    if render_show:
                        environment.render(savefig = render_save, 
                                           filename = fn)
                q_values_list.append(q_values)
                state_list.append(state)
                action_list.append(action)
            # Results summary for current data generator
            tot_rew = 0.
            if len(reward_list) > 0:
                tot_rew = round(sum(reward_list), 2)
            Logger.info("Gen-id: %d, reward: %s" % (ii, str(tot_rew)))
            Logger.info("Show env.step info: %s" % str(info))
            Logger.info(" * * * * * * * * * * * * ")
            
            trades_df = pd.DataFrame(trade_list) # Saeed 版本交易對帳單
            action_policy_df = pd.DataFrame({"q_values": q_values_list, 
                                             "state": state_list, 
                                             "action": action_list})
            reward_df = pd.DataFrame(reward_list)
            if save_results:
                # 個人版本新增
                environment.export_trade_records(fn) # 建倉平倉交易紀錄
                environment.export_asset_records(fn) # 資產水位記錄
                # 沿用自 Saeed 版本但更改檔名規則
                f1 = myEnv.Trade_Path + '\\' + fn + "_calc_return.csv"
                f2 = myEnv.Trade_Path + '\\' + fn + "_act_policy.pkl"
                f3 = myEnv.Trade_Path + '\\' + fn + "_reward.csv"
                if trades_df.shape[0] > 0:
                    trades_df.to_csv(f1)
                if action_policy_df.shape[0] > 0:
                    action_policy_df.to_pickle(f2) # pickling (模組化)
                if reward_df.shape[0] > 0:
                    reward_df.to_csv(f3)
        # End for loop for genlist
    except KeyboardInterrupt: # interrupted by user
        Logger.error("Keyboard interrupt ocurred! Task terminated ...")
        Logger.error("Agent testing does not complete!")
    except Exception as e:
        Logger.critical("Unexpected exceptions occurred!")
        Logger.critical("{}".format(e))
    
    Logger.info("Return the results from the last data generator.")
    print("Testing complete!")
    print("Return the results from the last data generator.")
    return({"trades_df": trades_df,
            "action_policy_df": action_policy_df,
            "reward_list": reward_list})

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def Batch_Describe_Performance(fnamelist):
    """
    myAna.Describe_Performance方法的批次處理版本
    """
    perf_batch = {}
    is_first_run = True
    for fname in fnamelist:
        apath = myEnv.Trade_Path + '\\' + fname + "_asset_rec.csv"
        tpath = myEnv.Trade_Path + '\\' + fname + "_trade_rec.csv"
        try:
            # 若計算過程被意外中止，則 apath 對應的檔案可能不存在或是空的
            asset_df = pd.read_csv(apath, index_col = None, \
                                   delimiter = myParams.Delimiter, \
                                   skiprows = None, header = 0)
        except Exception as e:
            Logger.warning("File %s does not exist or it is empty!" % apath)
            Logger.warning("Please check the running status!")
            Logger.warning("Exception messages as below:")
            Logger.warning("{}".format(e))
            asset_df = None
        try:
            # 若在 episode 終止前無交易記錄，則 tpath 對應的檔案可能不存在或是空的
            # 空檔案讀取例外：UnboundLocalError
            # 檔案不存在：FileNotFoundError
            trade_df = pd.read_csv(tpath, index_col = None, \
                                   delimiter = myParams.Delimiter, \
                                   skiprows = None, header = 0)
        except Exception as e:
            Logger.warning("File %s does not exist or it is empty!" % tpath)
            Logger.warning("Please check env status info or trade record!")
            Logger.warning("Exception messages as below:")
            Logger.warning("{}".format(e))
            trade_df = None
        # 若兩個檔案都存在，則評估各筆資料對應的績效指標
        if (not asset_df is None) and (not trade_df is None):
            Logger.debug("Evaluate performance ...")
            perf_tmp = myAna.Describe_Performance(asset_df, trade_df)
            if is_first_run:
                # Initialize perf_batch
                perf_batch["test_fname"] = [fname] # listing
                for key in perf_tmp.keys():
                    perf_batch[key] = [perf_tmp[key]] # copy by listing
                is_first_run = False
            else:
                perf_batch["test_fname"].append(fname)
                for key in perf_tmp.keys():
                    perf_batch[key].append(perf_tmp[key])
        # 清除前一次暫存的內容
        if not is_first_run: 
            for key in perf_tmp.keys():
                perf_tmp[key] = "NA"
    
    # 目前總共有23個分析欄位，如下
    try:
        perf_df = pd.DataFrame({"test_fname": perf_batch["test_fname"], 
                                "test_days": perf_batch["test_days"], 
                                "initial_cash": perf_batch["initial_cash"], 
                                "reward": perf_batch["reward"], 
                                "max_underwater_time": perf_batch["max_underwater_time"], 
                                "max_underwater_ratio": perf_batch["max_underwater_ratio"], 
                                "max_drawdown_ratio": perf_batch["max_drawdown_ratio"], 
                                "return": perf_batch["return"], 
                                "annual_return": perf_batch["annual_return"], 
                                "max_return": perf_batch["max_return"], 
                                "min_return": perf_batch["min_return"], 
                                "annual_volatility": perf_batch["annual_volatility"], 
                                "sharpe": perf_batch["sharpe"], 
                                "sortino": perf_batch["sortino"], 
                                "max_hold_days": perf_batch["max_hold_days"], 
                                "no_trades": perf_batch["no_trades"], 
                                "avg_hold_days": perf_batch["avg_hold_days"], 
                                "cover_ratio": perf_batch["cover_ratio"], 
                                "max_profit": perf_batch["max_profit"], 
                                "min_profit": perf_batch["min_profit"], 
                                "avg_profit": perf_batch["avg_profit"], 
                                "gain_pain_ratio": perf_batch["gain_pain_ratio"], 
                                "win_rate": perf_batch["win_rate"]})
        # 存檔記錄(於SaMaMind目錄下)
        r_path = RefPath + '\\' + "samamind_performance.xlsx"
        perf_df.to_excel(r_path, index = False, header = True)
        # 結果視覺化 (選擇數值類型欄位進行分析)
        perf_df_new = perf_df.drop(columns = ["test_fname", 
                                              "test_days", 
                                              "initial_cash"])
        p_path = RefPath + '\\' + "samamind_performance"
        try:
            assert perf_df_new.shape[0] >= 5
            myAna.Show_Perf_Dist(perf_df_new, 
                                 mode = "hist", 
                                 verbose = True, 
                                 save_path = p_path + "_hist.png")
            myAna.Show_Perf_Dist(perf_df_new, 
                                 mode = "violin", 
                                 verbose = False, 
                                 save_path = p_path + "_violin.png")
            Logger.debug("Finish plotting performance distribution of raw data!")
        except:
            Logger.info("Not enough raw data for visualization!")
        # Filter out rows with no. traders <= 2, then replot again
        # Current ver. only supports single filter condition
        # (Actually it can be extended thanks to the powerful pandas capability)
        try: 
            fil_col = "no_trades"
            thres_val = 2
            assert perf_df_new[perf_df_new[fil_col] > thres_val].shape[0] >= 5
            myAna.Show_Perf_Dist(perf_df_new[perf_df_new[fil_col] > thres_val], 
                                 mode = "hist", 
                                 verbose = True, 
                                 save_path = p_path + "_hist_filtered.png")
            myAna.Show_Perf_Dist(perf_df_new[perf_df_new[fil_col] > thres_val], 
                                 mode = "violin", 
                                 verbose = False, 
                                 save_path = p_path + "_violin_filtered.png")
            Logger.debug("Finish plot perf. distribution of filtered data!")
        except:
            Logger.info("Not enough filtered data for visualization!")
    except Exception as e:
        Logger.error("Problem encountered during exporting results!")
        Logger.error("Probably due to no performance evaluated!")
        Logger.error("Exception messages as below:")
        Logger.error("{}".format(e))
        perf_df = None
    return perf_df

# Function testing (myEnv + myAgent + myWorld + myAna + myDNNVisualizer)
if __name__ == "__main__":
    t_start = dt.datetime.now() # 計時
    print("***** Module myWorld / myEnv / myAgent test: *****")
    print("***** Auxiliary: myWorldSettings / myDNNVisualizer *****")
    # 參數設定
    # kwlist = list(myParams.DefinedTechInd)
    td = TradePeriod # from myPostProcessing module
    kwlist = ["MACD", "SMA", "CVOL"]
    # 取得後處理資料所需調用的欄位名稱
    # (需與 myPostProcessing 中設定一致)
    myEnv.GetColNames(td = td, kwlist = kwlist)
    paramset = ChangeTheWorld() # Use default
    
    # 模型訓練：
    """
    # 這裡需要設定的項目有
    # filename, symbol, save_results, agent_name, 
    # 其他由全域變數代入
    model_name = "model_test"
    fname = 5 # 可以是整數、None、或檔案路徑
    # fname = None # 可以是整數、None、或檔案路徑
    is_results_saved = True
    test_agent = myAgent.Chosen_Agent
    print("Start training ...")
    results_df = Training(filename = fname, 
                          symbol = model_name, 
                          save_results = is_results_saved, 
                          agent_name = test_agent, 
                          time_fee = Time_Fee, 
                          trading_fee = Trading_Fee, 
                          trading_tax = Trading_Tax, 
                          train_test_split = Train_Test_Split, 
                          episodes = Episodes, 
                          episode_length = Episode_Length, 
                          history_length = History_Length, 
                          memory_size = myAgent.Memory_Size, 
                          batch_size = myAgent.Batch_Size)
    print("Visualize training results ...")
    myAna.Training_Visualizer(results_df)
    print("Finish training ...")
    """
    """
    # load training record
    results_df = pd.read_csv(RefPath + "\\model_test_training.csv")
    myAna.Training_Visualizer(results_df)
    """
    
    # 模型測試
    """
    # 這裡需要設定的參數有:
    # filename, symbol, save_results, agent_name, train_test_split, 
    # save_results, render_save, render_show, search_folder
    model_name = "model_test"
    fname = 3 # 可以是整數、None、或檔案路徑
    # fname = None # 可以是整數、None、或檔案路徑
    is_results_saved = True
    test_agent = myAgent.Chosen_Agent
    
    search_dir = "test"
    # tts = Train_Test_Split
    tts = 0.0
    is_render_saved = True
    is_render_shown = True
    print("Start testing ...")
    results_df_dict = Testing(filename = fname, 
                              search_folder = search_dir, 
                              symbol = model_name, 
                              agent_name = test_agent, 
                              train_test_split = tts, 
                              save_results = is_results_saved, 
                              render_save = is_render_saved, 
                              render_show = is_render_shown, 
                              history_length = History_Length, 
                              time_fee = Time_Fee, 
                              trading_fee = Trading_Fee, 
                              trading_tax = Trading_Tax)
    print("Finish testing ...")
    """
    
    # 模型分析
    """
    fname_vis = "2330.TW_demo_v1" # 待分析的單一檔案名稱
    apath = myEnv.Trade_Path + '\\' + fname_vis + "_asset_rec.csv"
    tpath = myEnv.Trade_Path + '\\' + fname_vis + "_trade_rec.csv"
    asset_df = pd.read_csv(apath, index_col = None, \
                           delimiter = myParams.Delimiter, \
                           skiprows = None, header = 0)
    trade_df = pd.read_csv(tpath, index_col = None, \
                           delimiter = myParams.Delimiter, \
                           skiprows = None, header = 0)
    myAna.Asset_Visualizer(asset_df)
    perf = myAna.Describe_Performance(asset_df, trade_df)
    print("Trade performance summary:")
    print("***   ***")
    for key, value in perf.items():
        print("Indicator {k} = {v}".format(k = key, v = value))
    print("***   ***")
    """
    
    # 批次模型分析
    """
    lookup_folder = "test"
    fpathlist = GetFpathList(lookup_folder)
    fnamelist = []
    for fpath in fpathlist:
        # Extract filename in fplist via regexp:
        # fname = re.findall(r'[^\\]+(?=\.csv$)', fpath)[-1]
        fname = re.findall(r'[^\\]+(?=\.)', fpath)[-1]
        fnamelist.append(fname)
    perf_df = Batch_Describe_Performance(fnamelist)
    """
    
    # 模型視覺化 (使用 myDNNVisualizer 模組)
    model_name = "model_test"
    test_agent = myAgent.Chosen_Agent
    
    print("Visualize the RL model:")
    mf = model_name + '_' + test_agent + '.h5'
    mpath = myAgent.Model_Path
    f_path = mpath + '\\' + mf
    dnn = vis.Load_Model(f_path) # 取得 RL agent 模型資料
    dnn.summary() # 模型摘要
    vis.Print_Model_Struct(dnn) # 模型摘要
    
    pydot_path = mpath + '\\' + model_name + '_' + test_agent + '.png'
    # 匯出模型架構圖
    vis.Export_Model_Struct(dnn, to_file = pydot_path, \
                            show_shapes = True, \
                            show_layer_names = True)

    vis_path = mpath + '\\' + model_name + '_' + test_agent  + "_network.png"
    vis_path_1 = mpath + '\\' + model_name + '_' + test_agent + "_layer.png"
    ins = dnn.get_weights()[0].shape[0]
    outs = dnn.get_weights()[-1].shape[0]
    innames = ["In-" + str(i+1) for i in range(ins)]
    outnames = ["Out-" + str(i+1) for i in range(outs)]
    dnncoef = dnn.get_weights() # 模型權重資訊
    vis.VisualizeDNN(dnncoef, innames, outnames, vis_path) # DNN網路視覺化
    vis.VisualizeNetwork(dnncoef[0], vis_path_1) # 熱力圖分析單層權重數值分布
    
    t_end = dt.datetime.now()
    print("Total ellapsed time is: ", t_end - t_start)
# Done!
