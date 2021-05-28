# -*- coding: utf-8 -*-

import os
import datetime as dt
import math
import numpy as np
import pandas as pd
import re
import unittest

from myUtilities import myLog
from myMind import myEnv, myAgent
from myMind.myAgent import DQNAgent, DDQNAgent, DDDQNAgent
from myMind.myEnv import TAStreamer, Indicator_1

# 測試項目篩選設定
defaultReason = "Assigned by tester"
skip_TestEnv = [False, defaultReason]
skip_TestTAStreamer = [False, defaultReason]
skip_TestIndicator1 = [False, defaultReason]
skip_TestAgents = [True, defaultReason]

# 存取檔案的預設父路徑
defaultPath = os.path.dirname(os.getcwd())

# 設定單元測試日誌
# LogName = None # root logger
LogName = "myRL_Unittest"
Logger = myLog.Log(myLog.LogLevel.DEBUG, LogName)

# 測試前設置：
Dpath = ''
if __name__ == "__main__":
    Dpath = os.path.dirname(os.getcwd())
else:
    modulepath = os.path.dirname(__file__)
    Dpath = os.path.dirname(modulepath)

FilePath = Dpath + "\\2330.TW_demo_v1.xlsx" 
DF0 = pd.read_excel(FilePath, header = 0, index_col = None, skiprows = None)
DF = pd.DataFrame(data = [], columns = ['V', 'O', 'H', 'L', 'C'], \
                  index = DF0["Date"].values)
DF[list(DF.columns)] = DF0[list(DF.columns)].values
DF.index = pd.to_datetime(DF.index)

# --- --- #

@unittest.skipIf(skip_TestEnv[0], skip_TestEnv[1])
class TestEnv(unittest.TestCase):
    """ 進行myEnv模組中輔助函式操作測試 """
    def setUp(self):
        Logger.debug("Setup: TestEnv.")
        self.initial_cash = 10
        self.time_fee = 0.01
        self.t_hold = 30
    def tearDown(self):
        Logger.debug("Tearing down the TestEnv test.")
        myEnv.EnvReset()
        myEnv.Cols_1 = []
        myEnv.Cols_2 = []
    def test_env_reset(self):
        Logger.debug(" *** test_env_reset *** ")
        myEnv.EnvReset()
        self.assertEqual(myEnv.Initial_Cash, myEnv.Default_Initial_Cash)
    def test_env_setup(self):
        Logger.debug(" *** test_env_setup *** ")
        myEnv.EnvSetUp(spread = myEnv.Spread, 
                       prep_isreturnbasis = myEnv.Prep_IsReturnBasis, 
                       initial_cash = self.initial_cash, 
                       max_underwater_ratio = myEnv.Max_Underwater_Ratio, 
                       keep_cash_ratio = myEnv.Keep_Cash_Ratio, 
                       unit_quantity = myEnv.Unit_Quantity, 
                       hold_period_upper = myEnv.Hold_Period_Upper)
        self.assertTrue(myEnv.Default_Initial_Cash > myEnv.Initial_Cash)
    def test_get_colnames(self):
        Logger.debug(" *** test_get_colnames *** ")
        myEnv.GetColNames()
        Logger.debug("Cols_1: {}".format(' '.join(map(str, myEnv.Cols_1))))
        Logger.debug("Cols_2: {}".format(' '.join(map(str, myEnv.Cols_2))))
        self.assertTrue(len(myEnv.Cols_1) > 0)
        self.assertTrue(len(myEnv.Cols_2) > 0)
    def test_time_fee_penalty(self):
        Logger.debug(" *** test_time_fee_penalty *** ")
        ptf = myEnv.Penaltize_Time_Fee(time_fee = self.time_fee, 
                                       t_hold = self.t_hold)
        Logger.debug("t_hold = 30, time_fee = 0.01, penaltized = %s" % str(ptf))
        self.assertTrue(ptf > self.time_fee)


@unittest.skipIf(skip_TestTAStreamer[0], skip_TestTAStreamer[1])
class TestTAStreamer(unittest.TestCase):
    """
    進行myEnv模組中TAStreamer類別操作測試
    但會借用Indicator_1類別部分功能，以讀取TAStreamer生成之資料
    """
    def setUp(self):
        Logger.debug("Setup: TestTAStreamer.")
        self.fpath = FilePath
        self.train_test = "train"
        self.train_test_split = 0.8
        self.spread = 0.005
        self.isreturnbasis = False # price and indicator basis features
        self.test_step_max = 10
    def tearDown(self):
        Logger.debug("Tearing down the TestTAStreamer test.")
    def test_generator_train(self):
        Logger.debug(" *** test_generator_train *** ")
        myEnv.GetColNames(20, []) # no technical indicators applied
        generator = TAStreamer(fpath = self.fpath, 
                               mode = self.train_test, 
                               split = self.train_test_split, 
                               spread = self.spread, 
                               prep_isreturnbasis = self.isreturnbasis)
        eplen = math.floor(int(len(pd.read_excel(FilePath)) * \
                          self.train_test_split) / 10) * 10
        n = min(eplen, self.test_step_max)
        env = Indicator_1(data_generator = generator, episode_length = eplen)
        _ = env.reset()
        for _ in range(n): # dummy actions for iteration of generator
            _, _, _, _ = env.step(np.array([1, 0, 0])) # hold
        history_data = env.return_prices_history()
        if len(history_data) > 0:
            for i in range(len(history_data)):
                Logger.debug("{}".format(' '.join(map(str, history_data[i]))))
        self.assertTrue(len(history_data) > 0)
        self.assertTrue(len(history_data) > n) # since env._history_length > 1
    def test_generator_test(self):
        Logger.debug(" *** test_generator_test *** ")
        # Modify some variables
        self.train_test = "test"
        self.isreturnbasis = True # return basis features
        myEnv.GetColNames(20, ["MACD", "SMA", "CVOL"])
        generator = TAStreamer(fpath = self.fpath, 
                               mode = self.train_test, 
                               split = self.train_test_split, 
                               spread = self.spread, 
                               prep_isreturnbasis = self.isreturnbasis)
        eplen = math.floor(int(len(pd.read_excel(FilePath)) * \
                          (1. - self.train_test_split)) / 10) * 10
        n = min(eplen, self.test_step_max)
        env = Indicator_1(data_generator = generator, episode_length = eplen)
        _ = env.reset()
        for _ in range(n): # dummy actions for iteration of generator
            _, _, _, _ = env.step(np.array([1, 0, 0])) # hold
        history_data = env.return_prices_history()
        if len(history_data) > 0:
            for i in range(len(history_data)):
                Logger.debug("{}".format(' '.join(map(str, history_data[i]))))
        self.assertTrue(len(history_data) > 0)
        self.assertTrue(len(history_data) > n) # since env._history_length > 1


@unittest.skipIf(skip_TestIndicator1[0], skip_TestIndicator1[1])
class TestIndicator1(unittest.TestCase):
    """
    進行myEnv模組中Indicator_1類別操作測試
    使用random agent測試，涵蓋myWorld.Testing方法中多數功能
    """
    def setUp(self):
        Logger.debug("Setup: TestIndicator1.")
        myEnv.GetColNames(20, ["MACD", "SMA", "CVOL"])
        # myEnv.GetColNames(20, []) # no technical indicators applied
        self.fname = re.findall(r'[^\\]+(?=\.)', FilePath)[-1]
        self.gen = TAStreamer(fpath = FilePath, 
                              mode = "test", 
                              split = 0., 
                              spread = myEnv.Spread, 
                              prep_isreturnbasis = False)
        self.ep_len = math.floor(int(len(pd.read_excel(FilePath)) * 1.) \
                                 / 10) * 10
        # self.his_len = 2 # default
        self.his_len = 6 # 2021.02.26
        self.action_size = 0
        self.state_size = 0
        self.test_step_max = 10 # max steps for testing
        self.trading_fee = 0.001425
        self.trading_tax = 0.003
        self.time_fee = 0.01/365
    def tearDown(self):
        Logger.debug("Tearing down the TestIndicator1 test.")
    def test_init_env(self):
        Logger.debug(" *** test_init_env *** ")
        env = Indicator_1(data_generator = self.gen, 
                          episode_length = self.ep_len, 
                          history_length = self.his_len, 
                          trading_fee = self.trading_fee, 
                          trading_tax = self.trading_tax, 
                          time_fee = self.time_fee)
        self.action_size = env.get_action_size()
        state = env.reset()
        self.state_size = len(state)
        Logger.debug("action_size = %d" % self.action_size)
        Logger.debug("state_size = %d" % self.state_size)
        Logger.debug("states observed: {}".format(' '.join(map(str, state))))
        self.assertTrue(self.action_size > 0)
        self.assertTrue(self.state_size > 0)
    def test_random_agent(self):
        Logger.debug(" *** test_random_agent *** ")
        env = Indicator_1(data_generator = self.gen, 
                          episode_length = self.ep_len, 
                          history_length = self.his_len, 
                          trading_fee = self.trading_fee, 
                          trading_tax = self.trading_tax, 
                          time_fee = self.time_fee)
        self.action_size = env.get_action_size()
        state = env.reset()
        self.state_size = len(state)
        for _ in range(self.test_step_max):
            action = env.random_action_fun() # call random agent
            state, reward, done, info = env.step(action, True)
            Logger.debug("Step: %s, get reward: %s" % (str(_), str(reward)))
            # Check the output results for his_len > 2:
            history_data = env.return_prices_history()
            if len(history_data) > 0:
                for i in range(len(history_data)):
                    Logger.debug("{}".format(' '.join(map(str, \
                                                          history_data[i]))))
        Logger.debug("states observed: {}".format(' '.join(map(str, state))))
        self.assertTrue(len(action) > 0)
        self.assertIsInstance(reward, float)
    def test_export_record(self):
        Logger.debug(" *** test_export_record *** ")
        Logger.debug("Show filename: %s" % self.fname)
        env = Indicator_1(data_generator = self.gen, 
                          episode_length = self.ep_len, 
                          history_length = self.his_len, 
                          trading_fee = self.trading_fee, 
                          trading_tax = self.trading_tax, 
                          time_fee = self.time_fee)
        self.action_size = env.get_action_size()
        state = env.reset()
        self.state_size = len(state)
        done = False
        state_list = []
        action_list = []
        reward_list = []
        trade_list = []
        render_show = False
        render_save = False
        while not done:
            action = env.random_action_fun() # call random agent
            state, reward, done, info = env.step(action, True)
            if "status" in info and info["status"] == "Closed plot":
                done = True
            else:
                reward_list.append(reward)
                calc_returns = env.return_calc(render_show)
                # 運算結果即時輸出與顯示
                if calc_returns:
                    trade_list.append(calc_returns)
                if render_show: # set to false, do not rendering
                    env.render(savefig = render_save, filename = self.fname)
            state_list.append(state)
            action_list.append(action)
        Logger.debug("Show info: ")
        for key, value in info.items():
            Logger.debug("{k} = {v}".format(k = key, v = value))
        trades_df = pd.DataFrame(trade_list) # Saeed 版本交易對帳單
        action_policy_df = pd.DataFrame({"state": state_list, 
                                         "action": action_list})
        reward_df = pd.DataFrame(reward_list)
        env.export_trade_records(self.fname) # 建倉平倉交易紀錄
        env.export_asset_records(self.fname) # 資產水位記錄
        f1 = myEnv.Trade_Path + '\\' + self.fname + "_calc_return.csv"
        f2 = myEnv.Trade_Path + '\\' + self.fname + "_act_policy.pkl"
        f3 = myEnv.Trade_Path + '\\' + self.fname + "_reward.csv"
        Logger.debug("Size of trades_df: %d" % trades_df.shape[0])
        Logger.debug("Size of action_policy_df: %d" % \
                     action_policy_df.shape[0])
        Logger.debug("Size of reward_df: %d" % reward_df.shape[0])
        if trades_df.shape[0] > 0:
            trades_df.to_csv(f1)
        action_policy_df.to_pickle(f2) # pickling (模組化)
        if reward_df.shape[0] > 0:
            reward_df.to_csv(f3)
        self.assertTrue(action_policy_df.shape[0] > 0)
        self.assertEqual(done, True)
        self.assertIsInstance(info, dict)
    def test_render_show(self):
        Logger.debug(" *** test_render_show *** ")
        env = Indicator_1(data_generator = self.gen, 
                          episode_length = self.ep_len, 
                          history_length = self.his_len, 
                          trading_fee = self.trading_fee, 
                          trading_tax = self.trading_tax, 
                          time_fee = self.time_fee)
        self.action_size = env.get_action_size()
        state = env.reset()
        self.state_size = len(state)
        done = False
        render_show = True
        render_save = True
        while not done:
            action = env.random_action_fun() # call random agent
            state, reward, done, info = env.step(action, True)
            if "status" in info and info["status"] == "Closed plot":
                done = True
            else:
                if render_show:
                    env.render(savefig = render_save, filename = self.fname)
        Logger.debug("Show info: ")
        for key, value in info.items():
            Logger.debug("{k} = {v}".format(k = key, v = value))
        self.assertEqual(done, True)
        self.assertIsInstance(info, dict)


@unittest.skipIf(skip_TestAgents[0], skip_TestAgents[1])
class TestAgents(unittest.TestCase):
    """
    進行myAgent模組操作測試
    包含 DQNAgent / DDQNAgent / DDDQNAgent 與輔助函式
    涵蓋myWorld.Training方法中多數功能
    """
    def setUp(self):
        Logger.debug("Setup: TestAgents.")
        self.agent = None
        self.ep = 10 # episodes
        self.mem_size = 128
        self.batch_size = 16
        myEnv.GetColNames(20, []) # no technical indicators applied
        self.gen = TAStreamer(fpath = FilePath, 
                              mode = "train", 
                              split = 0.8, 
                              spread = myEnv.Spread, 
                              prep_isreturnbasis = False)
        self.ep_len = math.floor(int(len(pd.read_excel(FilePath)) * 0.8) \
                                 / 10) * 10
        self.env = Indicator_1(data_generator = self.gen, 
                               episode_length = self.ep_len, 
                               trading_fee = 0.001425, 
                               trading_tax = 0.003, 
                               time_fee = 0.01/365)
        self.action_size = self.env.get_action_size()
        state = self.env.reset()
        self.state_size = len(state)
    def tearDown(self):
        Logger.debug("Tearing down the TestAgents test.")
        myAgent.AgentReset()
    def test_agent_reset(self):
        Logger.debug(" *** test_agent_reset *** ")
        myAgent.AgentReset()
        self.assertEqual(myAgent.Memory_Size, myAgent.Default_Memory_Size)
        self.assertEqual(myAgent.Batch_Size, myAgent.Default_Batch_Size)
    def test_agent_setup(self):
        Logger.debug(" *** test_agent_setup *** ")
        myAgent.AgentSetUp(neurons_per_layer = myAgent.Neurons_Per_Layer, 
                           hiddenlayer_shape = myAgent.HiddenLayer_NParams_Ratio, 
                           chosen_agent = myAgent.Chosen_Agent, 
                           epsilon_min = myAgent.Epsilon_Min, 
                           gamma = myAgent.Gamma, 
                           activation = myAgent.Activation, 
                           l2_strength = myAgent.L2_Strength, 
                           learning_rate = myAgent.Learning_Rate, 
                           train_interval = myAgent.Train_Interval, 
                           update_target_freq = myAgent.Update_Target_Freq, 
                           memory_size = self.mem_size, 
                           batch_size = self.batch_size, 
                           validation_split = myAgent.Validation_Split)
        self.assertTrue(myAgent.Default_Memory_Size > myAgent.Memory_Size)
        self.assertTrue(myAgent.Default_Batch_Size > myAgent.Batch_Size)
    def test_train_agent(self):
        Logger.debug(" *** test_train_agent *** ")
        for i in range(len(myAgent.Agent_List)):
            # 使用子測試方法，測試不同的RL agent
            with self.subTest():
                agent_name = myAgent.Agent_List[i]
                Logger.debug("Subtest for %s." % agent_name)
                if agent_name == "DQN":
                    self.agent = DQNAgent(state_size = self.state_size, 
                                          action_size = self.action_size, 
                                          episodes = self.ep, 
                                          episode_length = self.ep_len, 
                                          no_generators = 1, 
                                          memory_size = self.mem_size, 
                                          batch_size = self.batch_size, 
                                          train_test = "train", 
                                          symbol = "unittest")
                elif agent_name == "DDQN":
                    self.agent = DDQNAgent(state_size = self.state_size, 
                                           action_size = self.action_size, 
                                           episodes = self.ep, 
                                           episode_length = self.ep_len, 
                                           no_generators = 1, 
                                           memory_size = self.mem_size, 
                                           batch_size = self.batch_size, 
                                           train_test = "train", 
                                           symbol = "unittest")
                elif agent_name == "DDDQN":
                    self.agent = DDDQNAgent(state_size = self.state_size, 
                                            action_size = self.action_size, 
                                            episodes = self.ep, 
                                            episode_length = self.ep_len, 
                                            no_generators = 1, 
                                            memory_size = self.mem_size, 
                                            batch_size = self.batch_size, 
                                            train_test = "train", 
                                            symbol = "unittest")
                else:
                    Logger.error("Undefined agent name %s!" % agent_name)
                # Warming up the agent with random action (full exploration)
                # 配置 memory，但 agent.observe() 還不會開始訓練模型
                done = False
                info = {}
                state = self.env.reset()
                counts = 0
                for _ in range(self.ep_len):
                    if counts >= self.mem_size:
                        Logger.debug("Memory refilled!")
                        break
                    else:
                        counts += 1
                    if done:
                        Logger.debug("Show env.step returned info: ")
                        for key, value in info.items():
                            Logger.debug("{k} = {v}".\
                                         format(k = key, v = value))
                        Logger.debug("Restart env now ...")
                        state = self.env.reset()
                        # break
                    action = self.agent.act(state)
                    next_state, reward, done, info = self.env.step(action, True)
                    self.agent.observe(state, action, reward, next_state, \
                                       done, warming_up = True)
                    # Logger.debug("Asset: {}".format(' '.join(map(str, self.env.asset_record[-1]))))
                Logger.debug("Target mem_size: %d, filled: %d" % \
                             (self.mem_size, counts))
                
                # Training:
                loss_list = []
                val_loss_list = []
                reward_list = []
                epsilon_list = []
                metrics_df = None
                for ep in range(self.ep):
                    Logger.debug("Current episode: %d" % ep)
                    state = self.env.reset()
                    rew = 0
                    loss_list_temp = []
                    val_loss_list_temp = []
                    info = {}
                    done = False
                    for _ in range(self.ep_len):
                        if done: # Episode finishes after previous iteration!
                            Logger.debug("Show env.step returned info: ")
                            for key, value in info.items():
                                Logger.debug("{k} = {v}".\
                                             format(k = key, v = value))
                            break # to next data generator
                        action = self.agent.act(state)
                        next_state, reward, done, info = \
                            self.env.step(action, True)
                        loss = self.agent.observe(state, action, reward, \
                                                  next_state, done)
                        state = next_state
                        rew += reward
                        if(loss):
                            loss_list_temp.append\
                                (round(loss.history["loss"][0], 4))
                            val_loss_list_temp.append\
                                (round(loss.history["val_loss"][0], 4))
                    loss_list.append(np.mean(loss_list_temp))
                    val_loss_list.append(np.mean(val_loss_list_temp))
                    reward_list.append(round(rew, 2))
                    epsilon_list.append(round(self.agent.epsilon, 4))
                    Logger.debug("Asset: {}".format(' '.join(map(str, self.env.asset_record[-1]))))
                Logger.info("Training completed! Save RL agent model ...")
                self.agent.save_model()
                metrics_df = pd.DataFrame({"loss": loss_list, 
                                           "val_loss": val_loss_list, 
                                           "reward": reward_list, 
                                           "epsilon":epsilon_list})
                mfname = Dpath + "\\unittest_train_" + agent_name + ".csv"
                metrics_df.to_csv(mfname)
                self.assertTrue(metrics_df.shape[0] > 0)
                self.assertEqual(done, True)
    def test_batch_train(self):
        Logger.debug(" *** test_batch_train *** ")
        gen_1 = TAStreamer(fpath = FilePath, 
                           mode = "train", 
                           split = 0.5, 
                           spread = myEnv.Spread, 
                           prep_isreturnbasis = False)
        gen_2 = TAStreamer(fpath = FilePath, 
                           mode = "test", 
                           split = 0.5, 
                           spread = myEnv.Spread, 
                           prep_isreturnbasis = False)
        genlist = [gen_1, gen_2]
        # Two generators have equal length!
        eplen = math.floor(int(len(pd.read_excel(FilePath)) * 0.5) / 10) * 10
        # Use DDDQN agent
        self.agent = DDDQNAgent(state_size = self.state_size, 
                                action_size = self.action_size, 
                                episodes = self.ep, 
                                episode_length = eplen, 
                                no_generators = len(genlist), 
                                memory_size = self.mem_size, 
                                batch_size = self.batch_size, 
                                train_test = "train", 
                                symbol = "unittest_batch")
        # Warming up the agent with random action (full exploration)
        # 配置 memory，但 agent.observe() 還不會開始訓練模型
        counts = 0
        t_s = dt.datetime.now() # 計時
        for i in range(len(genlist)):
            done = False
            info = {}
            env = Indicator_1(data_generator = genlist[i], 
                                   episode_length = eplen, 
                                   trading_fee = 0.001425, 
                                   trading_tax = 0.003, 
                                   time_fee = 0.01/365)
            state = env.reset()
            for _ in range(eplen):
                if counts >= self.mem_size:
                    Logger.debug("Memory refilled!")
                    break
                else:
                    counts += 1
                if done:
                    Logger.debug("Show env.step returned info: ")
                    for key, value in info.items():
                        Logger.debug("{k} = {v}".format(k = key, v = value))
                    break
                action = self.agent.act(state)
                next_state, reward, done, info = env.step(action, True)
                self.agent.observe(state, action, reward, next_state, \
                                   done, warming_up = True)
            Logger.debug("gen_id: %d counts: %d" % (i, counts))
        Logger.debug("Target mem_size: %d, filled: %d" % \
                     (self.mem_size, counts))
        t_e = dt.datetime.now()
        Logger.info("Ellapsed time for warm-up: %s" % str(t_e - t_s))
        
        # Training:
        loss_list = []
        val_loss_list = []
        reward_list = []
        epsilon_list = []
        metrics_df = None
        Logger.debug("# of data generators: %d" % len(genlist))
        # Batch file mode of Saeed's World function (the training part)
        t_s = dt.datetime.now() # 計時
        for ep in range(self.ep):
            Logger.debug("Current episode: %d" % ep)
            rew = 0
            loss_list_temp = []
            val_loss_list_temp = []
            # For each episode, run N generators
            for gen in genlist:
                info = {}
                done = False
                # Re-initialize and change the data generator
                env = Indicator_1(data_generator = gen, 
                                       episode_length = eplen, 
                                       trading_fee = 0.001425, 
                                       trading_tax = 0.003, 
                                       time_fee = 0.01/365)
                state = env.reset()
                for _ in range(eplen):
                    if done: # Episode finishes after previous iteration!
                        Logger.debug("Show env.step returned info: ")
                        for key, value in info.items():
                            Logger.debug("{k} = {v}".\
                                         format(k = key, v = value))
                        break # to next data generator
                    action = self.agent.act(state)
                    next_state, reward, done, info = env.step(action, True)
                    # Keras的fit方法會回傳 history loss，包含每個 epoch 的 loss
                    # 目前在 agent.observe() 中，參數 epochs = 1，故只會有一個值
                    # Below is Saeed's comment ...
                    # loss would be none if the episode length is not % by 10
                    loss = self.agent.observe(state, action, reward, \
                                              next_state, done)
                    state = next_state
                    rew += reward
                    if(loss):
                        loss_list_temp.append\
                            (round(loss.history["loss"][0], 4))
                        val_loss_list_temp.append\
                            (round(loss.history["val_loss"][0], 4))
            loss_list.append(np.mean(loss_list_temp))
            val_loss_list.append(np.mean(val_loss_list_temp))
            reward_list.append(round(rew, 2))
            epsilon_list.append(round(self.agent.epsilon, 4))
        Logger.info("Training completed! Save RL agent model ...")
        t_e = dt.datetime.now()
        Logger.info("Ellapsed time for training: %s" % str(t_e - t_s))
        self.agent.save_model()
        
        metrics_df = pd.DataFrame({"loss": loss_list, 
                                   "val_loss": val_loss_list, 
                                   "reward": reward_list, 
                                   "epsilon":epsilon_list})
        mfname = Dpath + "\\unittest_batch_train_DDDQN.csv"
        metrics_df.to_csv(mfname)
        self.agent.load_model() # test load model
        self.assertTrue(metrics_df.shape[0] > 0)
        self.assertEqual(done, True)


# --- --- #
if __name__ == "__main__":
    print(" ***** Module myRLTest ***** ")
    t_start = dt.datetime.now() # 計時
    if not skip_TestEnv[0]:
        print("Test myEnv module, all functions!")
    if not skip_TestTAStreamer[0]:
        print("Test myEnv.TAStreamer class!")
    if not skip_TestIndicator1[0]:
        print("Test myEnv.Indicator_1 class!")
    if not skip_TestAgents[0]:
        print("Test myAgent module, all classes and functions!")
    unittest.main()
    t_end = dt.datetime.now()
    print("Total ellapsed time is: ", t_end - t_start)
# Done!

