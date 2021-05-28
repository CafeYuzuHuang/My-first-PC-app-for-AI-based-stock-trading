# -*- coding: utf-8 -*-

import os
import numpy as np
import random
from tensorflow.config import list_physical_devices
from tensorflow.keras.layers import Dense, Lambda, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from myUtilities import myParams
from myUtilities import myLog


"""
改寫自 Saeed Rahman 等人的研究和程式碼
來源：https://github.com/saeed349/Deep-Reinforcement-Learning-in-Trading
myAgent涵蓋 Agent 模組中的所有功能

myAgent.py 所具有的功能
數個 deep reinforcement learning (DRL) 中所使用的 agents:
    deep Q-learning network (DQN)
    double deep Q-learning network (DDQN)
    double Dueling deep Q-learning network (DDDQN or D3QN)
與 Barto & Sutton 的模型不同，在此以 DNN 來近似 tabular Q-values
另外使用 experience replay (Lin, 1992) 來增加 DNN 訓練之穩定性
※ 目前版本所有的 agent 都是基於 greedy policy 操作，
"""

LogName = __name__ # 日誌名稱設為模組名稱
RefPath = '' # 參考路徑
if __name__ == "__main__": # executed
    RefPath = os.path.dirname(os.getcwd())
    LogName = None # 當此腳本被執行時為根日誌(不設名稱)
else: # imported
    modulepath = os.path.dirname(__file__)
    RefPath = os.path.dirname(modulepath)
Logger = myLog.Log(myParams.DefaultRootLogLevel, LogName)

# 將運算裝置寫入log
try:
    Logger.info(" ***  Physical devices for DRL computing  *** ")
    Logger.info('{}'.join(map(str, list_physical_devices("CPU"))))
    Logger.info('{}'.join(map(str, list_physical_devices("GPU"))))
    Logger.info(" ***  (A blank line above means no GPU used.)  *** \n")
except:
    Logger.warning("Call tf.config.list_physical_devices runs into trouble!")

Model_Path = RefPath + '\\' + myParams.AI_Foldername # 存放模型參數
if not os.path.isdir(Model_Path): os.mkdir(Model_Path)

# Global constants
Agent_List = ["DQN", "DDQN", "DDDQN"]
Train_Test = "train"
Symbol = ''
Act_List = ["sigmoid", "softmax", "tanh", "relu", "leakyrelu", "selu", "elu"]
LeakyAlpha = 0.3 # leaky slope, keras default = 0.3

# GradClip = "norm" # clip by norm
# GradClip = "value" # clip by value
GradClip = "none" # no clipping; default

BiasInitializer = "zeros" # default
KernelInitializer = "glorot_uniform" # default
# KernelInitializer = "he_normal"
# KernelInitializer = "lecun_normal"

Default_Neurons_Per_Layer = 24
Default_Activation = "relu"
Default_Agent = "DDDQN"
Default_Epsilon_Min = 0.01 # 0 <= epsilon <= 1
Default_L2_Strength = 0.01
Default_Train_Interval = 100
Default_Update_Target_Freq = 100
Default_Memory_Size = 2048
Default_Gamma = 0.95 # discount factor, 0 <= gamma <= 1
Default_Learning_Rate = 0.001
Default_Batch_Size = 64 # Minibatch大小，需小於Memory size
Default_Validation_Split = 0.2

# Global variables
# 以下變數與建立DNN相關，由函式直接調用
HiddenLayer_NParams_Ratio = [1, 2, 4] # 元素數目表示隱藏層數量
Neurons_Per_Layer = Default_Neurons_Per_Layer
Activation = Default_Activation
L2_Strength = Default_L2_Strength
# 以下變數由引數傳遞，在myWorld使用
Chosen_Agent = Default_Agent
# 以下變數由引數傳遞，與模型訓練相關，用於Keras之optimizer
Learning_Rate = Default_Learning_Rate
# 以下變數由引數傳遞，與模型訓練相關(Agent.observe方法調用)
# 若未來沒有於myWorld做設定值驗證之需求，則將其從類別/函式引數中移除
Memory_Size = Default_Memory_Size
Batch_Size = Default_Batch_Size
Epsilon_Min = Default_Epsilon_Min
Train_Interval = Default_Train_Interval
Update_Target_Freq = Default_Update_Target_Freq # DQN不會用到
Gamma = Default_Gamma # 折現率，數值愈小表示未來的不確定性愈高
Validation_Split = Default_Validation_Split

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def AgentReset():
    """ 還原初始設定 """
    global Neurons_Per_Layer, Activation, Chosen_Agent
    global Epsilon_Min, L2_Strength, Train_Interval, Update_Target_Freq
    global Memory_Size, Gamma, Learning_Rate, Batch_Size
    global HiddenLayer_NParams_Ratio, Validation_Split
    Neurons_Per_Layer = Default_Neurons_Per_Layer
    Activation = Default_Activation
    Chosen_Agent = Default_Agent
    Epsilon_Min = Default_Epsilon_Min
    L2_Strength = Default_L2_Strength
    Train_Interval = Default_Train_Interval
    Update_Target_Freq = Default_Update_Target_Freq
    Memory_Size = Default_Memory_Size
    Gamma = Default_Gamma
    Learning_Rate = Default_Learning_Rate
    Batch_Size = Default_Batch_Size
    HiddenLayer_NParams_Ratio = [1, 2, 4]
    Validation_Split = Default_Validation_Split
    return None

@myLog.Log_Error(myParams.DefaultErrorLogLevel)
@myLog.Log_Func(myParams.DefaultFuncLogLevel, None, None)
def AgentSetUp(**kwargs):
    """
    設定RL的agent，修改預設參數。已定義關鍵字：
    neurons_per_layer (int, [1, ) ), 
    activation (str, defined ), 
    chosen_agent (str, defined ), 
    epsilon_min (float, [0, 1] ), 
    l2_strength (float, [0, 1] ), 
    train_interval (int, [1, ) ), 
    update_target_freq (int, [1, ) ),
    memory_size (int, [1, ) ), 
    gamma, (float, (0, 1] ), 
    learning_rate (float, (0, 1) ), 
    batch_size, (int, [1, memory_size] ), 
    hiddenlayer_shape (list, all elements are int)
    validation_split (float, [0, 1) )
    """
    global Neurons_Per_Layer, Activation, Chosen_Agent
    global Epsilon_Min, L2_Strength, Train_Interval, Update_Target_Freq
    global Memory_Size, Gamma, Learning_Rate, Batch_Size
    global HiddenLayer_NParams_Ratio, Validation_Split
    AgentReset() # Reset
    
    if "neurons_per_layer" in kwargs:
        if int(kwargs["neurons_per_layer"]) >= 1: # validation
            Neurons_Per_Layer = int(kwargs["neurons_per_layer"])
        else:
            Logger.warning("Invalid neurons_per_layer! (required: >= 1.)")
    if "activation" in kwargs:
        if kwargs["activation"] in Act_List: # validation
            Activation = kwargs["activation"]
        else:
            Logger.warning("Undefined activation! Required: ")
            Logger.warning("{}".format(', '.join(map(str, Act_List))))
    if "chosen_agent" in kwargs:
        if kwargs["chosen_agent"] in Agent_List: # validation
            Chosen_Agent = kwargs["chosen_agent"]
        else:
            Logger.warning("Undefined chosen_agent! Required: ")
            Logger.warning("{}".format(', '.join(map(str, Agent_List))))
    if "epsilon_min" in kwargs:
        if kwargs["epsilon_min"] <= 1.0 and \
            kwargs["epsilon_min"] >= 0.0:
            Epsilon_Min = kwargs["epsilon_min"]
        else:
            Logger.warning("Invalid epsilon_min! (required: <= 1 and >= 0.)")
    if "l2_strength" in kwargs:
        if kwargs["l2_strength"] <= 1.0 and \
            kwargs["l2_strength"] >= 0.0:
            L2_Strength = kwargs["l2_strength"]
        else:
            Logger.warning("Invalid L2_strength! (required: <= 1 and >= 0.)")
    if "train_interval" in kwargs:
        if int(kwargs["train_interval"]) >= 1: # validation
            Train_Interval = int(kwargs["train_interval"])
        else:
            Logger.warning("Invalid train_interval! (required: >= 1.)")
    if "update_target_freq" in kwargs:
        if int(kwargs["update_target_freq"]) >= 1: # validation
            Update_Target_Freq = int(kwargs["update_target_freq"])
        else:
            Logger.warning("Invalid update_target_freq! (required: >= 1.)")
    if "memory_size" in kwargs: # prior to batch_size
        if int(kwargs["memory_size"]) >= 1: # validation
            Memory_Size = int(kwargs["memory_size"])
        else:
            Logger.warning("Invalid memory_size! (required: >= 1.)")
    if "batch_size" in kwargs:
        if int(kwargs["batch_size"]) >= 1 and \
            int(kwargs["batch_size"]) <= Memory_Size:
            Batch_Size = int(kwargs["batch_size"]) # validated
        else:
            Logger.warning("Invalid batch_size!")
            Logger.warning("Required: memory_size >= batch_size >= 1.")
            # Default_Batch_Size vs. current Memory_Size
            if Batch_Size > Memory_Size:
                Logger.warning("Batch size should be l.e. memory size!")
                Batch_Size = Memory_Size
    if "gamma" in kwargs:
        if kwargs["gamma"] <= 1.0 and kwargs["gamma"] > 0.0:
            Gamma = kwargs["gamma"] # validated
        else:
            Logger.warning("Invalid gamma! (Required: 1 >= gamma > 0.)")
    if "learning_rate" in kwargs:
        if kwargs["learning_rate"] < 1.0 and \
            kwargs["learning_rate"] > 0.0:
            Learning_Rate = kwargs["learning_rate"] # validated
        else:
            Logger.warning("Invalid learning_rate! (Required: 1 > LR > 0)")
    if "hiddenlayer_shape" in kwargs:
        # check list length and element data type:
        if type(kwargs["hiddenlayer_shape"]) is list:
            nn = len(kwargs["hiddenlayer_shape"])
            if nn > 0: 
                tt = [type(kwargs["hiddenlayer_shape"][i]) is int \
                      for i in range(nn)]
                if all(tt):
                    HiddenLayer_NParams_Ratio = kwargs["hiddenlayer_shape"]
                else:
                    Logger.warning("Invalid element type found!")
                    Logger.warning("Any in hiddenlayer_shape is not int.")
            else:
                Logger.warning("Empty list for hiddenlayer_shape!")
        else:
            Logger.warning("Invalid data type for hiddenlayer_shape!")
    if "validation_split" in kwargs:
        if kwargs["validation_split"] < 1.0 and \
            kwargs["validation_split"] >= 0.0:
            Validation_Split = kwargs["validation_split"]
        else:
            Logger.warning("Invalid validation_split!")
            Logger.warning("(required: < 1 and >= 0.)")
    # Examine the variables:
    Logger.info("neurons_per_layer: %s" % str(Neurons_Per_Layer))
    Logger.info("hiddenlayer_shape:")
    Logger.info("{}".format(', '.join(map(str, HiddenLayer_NParams_Ratio))))
    Logger.info("activation: %s" % Activation)
    Logger.info("chosen_agent: %s" % Chosen_Agent)
    Logger.info("epsilon_min: %s" % str(Epsilon_Min))
    Logger.info("l2_strength: %s" % str(L2_Strength))
    Logger.info("train_interval: %s" % str(Train_Interval))
    Logger.info("update_target_freq: %s" % str(Update_Target_Freq))
    Logger.info("memory_size: %s" % str(Memory_Size))
    Logger.info("batch_size: %s" % str(Batch_Size))
    Logger.info("gamma: %s" % str(Gamma))
    Logger.info("learning_rate: %s" % str(Learning_Rate))
    Logger.info("validation_split: %s" % str(Validation_Split))
    return None


class Agent:
    """ Abstract class for an agent. """
    def __init__(self, epsilon = None):
        """
        Agent initialization
        epsilon (optional): the exploration starting rate
        """
        self.epsilon = epsilon
    
    def act(self, state):
        """
        Action function.
        This function takes a state (from an environment) as an argument and
        returns an action.
        Arguments:
            state (numpy.array): state vector
        Returns:
            np.array: numpy array of the action to take
        """
        raise NotImplementedError()
    
    def observe(self, state, action, reward, next_state, terminal, *args):
        """
        Observe function.
        This function takes a state, a reward and a terminal boolean 
        and returns a loss value. This is only used for learning agents.
        Arguments:
            state (numpy.array): state vector
            action (numpy.array): action vector
            reward (float): reward value
            next_state (numpy.array): next state vector
            terminal (bool): whether the game is over or not
        Returns:
            float: value of the loss
        """
        raise NotImplementedError()
    
    def end(self):
        """ End of episode logic. """
        pass

class DQNAgent(Agent):
    """
    (Vanilla) deep Q-learning Network agent, comparative learned agent.
    Q*(s_t, a) = r(s_t, a) + gamma * max_a_Q*(s_t+1, a)
    若 epsilon_min = 1.0 則為 random agent
    """
    def __init__(self,
                 state_size,
                 action_size,
                 episodes,
                 episode_length,
                 no_generators, 
                 train_interval = Train_Interval,
                 update_target_freq = Update_Target_Freq,
                 memory_size = Memory_Size,
                 gamma = Gamma,
                 learning_rate = Learning_Rate,
                 batch_size = Batch_Size,
                 epsilon_min = Epsilon_Min,
                 train_test = Train_Test,
                 symbol = Symbol, 
                 validation_split = Validation_Split):
        """ Initialize a DQN agent """
        self.state_size = state_size
        self.action_size = action_size # buy, sell, hold
        self.memory_size = memory_size
        self.memory = [None] * memory_size
        self.gamma = gamma # discount factor
        self.epsilon = 1.0 # 100% exploration at beginning
        self.epsilon_min = epsilon_min
        # epsilon_decrement as linear decrease rate
        self.epsilon_decrement = (self.epsilon - epsilon_min) * \
            train_interval / (episodes * episode_length * no_generators)
        self.learning_rate = learning_rate
        self.train_interval = train_interval
        self.update_target_freq = update_target_freq
        self.batch_size = batch_size
        if Activation.lower() == "leakyrelu":
            self.brain = self._build_brain_2()
            # self.brain_ = self._build_brain_2()
        else:
            self.brain = self._build_brain()
            # self.brain_ = self._build_brain()
        self.i = 0
        self.train_test = train_test
        self.symbol = symbol # model name
        self.val_split = validation_split
        
    def save_model(self):
        """ Save the current DQN agent model """
        fname = self.symbol + "_DQN.h5"
        Logger.info("Save model: %s" % fname)
        self.brain.save(Model_Path + '\\' + fname)
        
    def load_model(self):
        """ Load the current DQN agent model """
        fname = self.symbol + "_DQN.h5"
        Logger.info("Load model: %s" % fname)
        # Apply Keras' load_model method
        self.brain = load_model(Model_Path + '\\' + fname)
        
    def _build_brain(self):
        """ Build the agent's brain (deep neural network) """
        neurons_per_layer = Neurons_Per_Layer
        activation = Activation
        l2str = L2_Strength
        brain = Sequential() # Apply Keras module; sequential DNN
        # 未來可調整以下DNN之設計，例如增加層數、改變網路形狀、
        # 添加 ridge regularization、修改 loss function, 
        # 使用 early stopping (callbacks) 或 drop-out 等
        # 以下對權重施加L2 norm (ridge regularization)
        HLNPR = HiddenLayer_NParams_Ratio
        brain.add(Dense(neurons_per_layer * HLNPR[0], 
                        input_dim = self.state_size, 
                        use_bias = True, 
                        kernel_initializer = KernelInitializer, 
                        bias_initializer = BiasInitializer, 
                        activation = activation, 
                        kernel_regularizer = l2(l2str))) # input layer
        for i in range(1, len(HLNPR)):
            brain.add(Dense(neurons_per_layer * HLNPR[i], 
                            use_bias = True, 
                            kernel_initializer = KernelInitializer, 
                            bias_initializer = BiasInitializer, 
                            activation = activation, 
                            kernel_regularizer = l2(l2str)))
        brain.add(Dense(self.action_size, 
                        use_bias = True, 
                        kernel_initializer = KernelInitializer, 
                        bias_initializer = BiasInitializer, 
                        activation = "linear")) # output layer
        if GradClip.lower() == "norm":
            brain.compile(loss = "mse", 
                          optimizer = Adam(lr = self.learning_rate, 
                                           clipnorm = 1.0))
        elif GradClip.lower() == "value":             
            brain.compile(loss = "mse", 
                          optimizer = Adam(lr = self.learning_rate, 
                                           clipvalue = 1.0))
        else: # default: no clipping
            brain.compile(loss = "mse", 
                          optimizer = Adam(lr = self.learning_rate))
        # 列印模型摘要
        brain.summary(print_fn = Logger.info) # change print handler
        return brain
    
    def _build_brain_2(self):
        """ Build the agent's brain (deep neural network) """
        # Relu 對於取得稀疏性有幫助，但可能會導致零梯度的產生(dying Relu problem)
        # 因此，若稀疏性不是那麼重要(非影像辨識問題須強調對比性)，相對
        # 我們會希望提升訓練能力並增加DNN深度，故適合改用leakyrelu
        neurons_per_layer = Neurons_Per_Layer
        l2str = L2_Strength
        brain = Sequential()
        HLNPR = HiddenLayer_NParams_Ratio
        brain.add(Dense(neurons_per_layer * HLNPR[0], 
                        input_dim = self.state_size, 
                        use_bias = True, 
                        kernel_initializer = KernelInitializer, 
                        bias_initializer = BiasInitializer, 
                        kernel_regularizer = l2(l2str))) # input layer
        brain.add(LeakyReLU(alpha = LeakyAlpha))
        for i in range(1, len(HLNPR)):
            brain.add(Dense(neurons_per_layer * HLNPR[i], 
                            use_bias = True, 
                            kernel_initializer = KernelInitializer, 
                            bias_initializer = BiasInitializer, 
                            kernel_regularizer = l2(l2str)))
            brain.add(LeakyReLU(alpha = LeakyAlpha))
        brain.add(Dense(self.action_size, 
                        use_bias = True, 
                        kernel_initializer = KernelInitializer, 
                        bias_initializer = BiasInitializer, 
                        activation = "linear")) # output layer
        if GradClip.lower() == "norm":
            brain.compile(loss = "mse", 
                          optimizer = Adam(lr = self.learning_rate, 
                                           clipnorm = 1.0))
        elif GradClip.lower() == "value":             
            brain.compile(loss = "mse", 
                          optimizer = Adam(lr = self.learning_rate, 
                                           clipvalue = 1.0))
        else: # default: no clipping
            brain.compile(loss = "mse", 
                          optimizer = Adam(lr = self.learning_rate))
        
        # 列印模型摘要
        brain.summary(print_fn = Logger.info) # change print handler
        return brain
    
    def act(self, state, test = False):
        """ Acting policy of the DQNAgent """
        act_values = []
        action = np.zeros(self.action_size)
        if np.random.rand() <= self.epsilon and \
            self.train_test == "train" and not test:
            # 從買/賣/持有(或空手)這三個動作中隨機選一個
            action[random.randrange(self.action_size)] = 1
        else:
            state = state.reshape(1, self.state_size)
            act_values = self.brain.predict(state)
            # 找出具有最高Q-value所對應的行動，將其設為1
            action[np.argmax(act_values[0])] = 1
        if test:
            return action, act_values
        else:
            return action
        
    def observe(self, state, action, reward, next_state, done, \
                warming_up = False):
        """ Memory management and training of the agent """
        self.i = (self.i + 1) % self.memory_size
        # Add arguments into memory array
        self.memory[self.i] = (state, action, reward, next_state, done)
        if (self.i == self.memory_size - 1):
            # Logger.info("Memory Refilled.")
            pass
        if (not warming_up) and (self.i % self.train_interval) == 0:
            if self.epsilon > self.epsilon_min: # decrease epsilon
                self.epsilon -= self.epsilon_decrement
                if self.epsilon < 0.:
                    self.epsilon = 0.
            state, action, reward, next_state, done = self._get_batches()
            # np.amax(axis = 1)找出所選擇的行動所對應的 Q value
            # 利用布林變數 done 來判斷哪個動作對應的 reward 需要更新
            reward += (self.gamma
                       * np.logical_not(done)
                       * np.amax(self.brain.predict(next_state),
                                 axis = 1))
            # brain.predict 回傳各動作的 Q value
            q_target = self.brain.predict(state)
            # 利用 action[1] 來判斷哪個動作對應的 Q value 需要更新
            # action[0] 對應的是 batch 中各組資料的 index
            q_target[action[0], action[1]] = reward
            return self.brain.fit(state, q_target,
                                  batch_size = self.batch_size,
                                  epochs = 1,
                                  verbose = False,
                                  validation_split = self.val_split)
        
    def _get_batches(self):
        """
        Selecting a batch of memory
        Split it into categorical subbatches
        Process action_batch into a position vector
        """
        # (Uniform) experience replay
        # For implementing PER, modify this function.
        
        # From memory array, sampling # = batch_size of data
        # e.g. sampling 64 batches (a minibatch) from 512 memory data
        batch = np.array(random.sample(self.memory, self.batch_size), \
                         dtype = object) # 2021.01.22 for ragged nested seq.
        
        state_batch = np.concatenate(batch[:, 0]) \
            .reshape(self.batch_size, self.state_size)
        action_batch = np.concatenate(batch[:, 1]) \
            .reshape(self.batch_size, self.action_size)
        reward_batch = batch[:, 2]
        next_state_batch = np.concatenate(batch[:, 3]) \
            .reshape(self.batch_size, self.state_size)
        done_batch = batch[:, 4]
        # Action processing
        # 取得執行動作對應的 batch index 與動作的項目
        action_batch = np.where(action_batch == 1)
        # 回傳 batch (or minibatch) 資訊
        return state_batch, action_batch, reward_batch, \
            next_state_batch, done_batch

class DDQNAgent(Agent):
    """
    Double Deep Q-learning Network agent, working learned agent.
    To avoid overestimate the Q function, two separated Q functions
    are learned independently:
    Q*(s_t, a) = r(s_t, a) + gamma * 
                 Q'*(s'_t+1, max_a_Q*(s_t+1', a'))
    Referred to DQNAgent class for comments in details!
    """
    def __init__(self,
                 state_size,
                 action_size,
                 episodes,
                 episode_length, 
                 no_generators, 
                 train_interval = Train_Interval,
                 update_target_freq = Update_Target_Freq,
                 memory_size = Memory_Size,
                 gamma = Gamma,
                 learning_rate = Learning_Rate,
                 batch_size = Batch_Size,
                 epsilon_min = Epsilon_Min,
                 train_test = Train_Test,
                 symbol = Symbol, 
                 validation_split = Validation_Split):
        """
        Initialize a DDQN agent
        Referred to DQNAgent class for comments in details.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.memory = [None] * memory_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decrement = (self.epsilon - epsilon_min) * \
            train_interval / (episodes * episode_length * no_generators)
        self.learning_rate = learning_rate
        self.train_interval = train_interval
        self.update_target_freq = update_target_freq
        self.batch_size = batch_size
        if Activation.lower() == "leakyrelu":
            self.brain = self._build_brain_2()
            self.brain_ = self._build_brain_2()
        else:
            self.brain = self._build_brain()
            self.brain_ = self._build_brain()
        self.i = 0
        self.train_test = train_test
        self.symbol = symbol
        self.val_split = validation_split
        
    def save_model(self):
        """ Save the current DDQN agent model """
        fname = self.symbol + "_DDQN.h5"
        Logger.info("Save model: %s" % fname)
        self.brain.save(Model_Path + '\\' + fname)
        
    def load_model(self):
        """ Load the current DDQN agent model """
        fname = self.symbol + "_DDQN.h5"
        Logger.info("Load model: %s" % fname)
        self.brain = load_model(Model_Path + '\\' + fname)
        
    def _build_brain(self):
        """ Build the agent's brain (deep neural network) """
        neurons_per_layer = Neurons_Per_Layer
        activation = Activation
        l2str = L2_Strength
        brain = Sequential()
        HLNPR = HiddenLayer_NParams_Ratio
        brain.add(Dense(neurons_per_layer * HLNPR[0], 
                        input_dim = self.state_size, 
                        use_bias = True, 
                        kernel_initializer = KernelInitializer, 
                        bias_initializer = BiasInitializer, 
                        activation = activation, 
                        kernel_regularizer = l2(l2str))) # input layer
        for i in range(1, len(HLNPR)):
            brain.add(Dense(neurons_per_layer * HLNPR[i], 
                            use_bias = True, 
                            kernel_initializer = KernelInitializer, 
                            bias_initializer = BiasInitializer, 
                            activation = activation, 
                            kernel_regularizer = l2(l2str)))
        brain.add(Dense(self.action_size, 
                        use_bias = True, 
                        kernel_initializer = KernelInitializer, 
                        bias_initializer = BiasInitializer, 
                        activation = "linear"))
        if GradClip.lower() == "norm":
            brain.compile(loss = "mse", 
                          optimizer = Adam(lr = self.learning_rate, 
                                           clipnorm = 1.0))
        elif GradClip.lower() == "value":             
            brain.compile(loss = "mse", 
                          optimizer = Adam(lr = self.learning_rate, 
                                           clipvalue = 1.0))
        else: # default: no clipping
            brain.compile(loss = "mse", 
                          optimizer = Adam(lr = self.learning_rate))
        # 列印模型摘要
        brain.summary(print_fn = Logger.info) # change print handler
        return brain
    
    def _build_brain_2(self):
        """ Build the agent's brain (deep neural network) """
        neurons_per_layer = Neurons_Per_Layer
        l2str = L2_Strength
        brain = Sequential()
        HLNPR = HiddenLayer_NParams_Ratio
        brain.add(Dense(neurons_per_layer * HLNPR[0], 
                        input_dim = self.state_size, 
                        use_bias = True, 
                        kernel_initializer = KernelInitializer, 
                        bias_initializer = BiasInitializer, 
                        kernel_regularizer = l2(l2str))) # input layer
        brain.add(LeakyReLU(alpha = LeakyAlpha))
        for i in range(1, len(HLNPR)):
            brain.add(Dense(neurons_per_layer * HLNPR[i], 
                            use_bias = True, 
                            kernel_initializer = KernelInitializer, 
                            bias_initializer = BiasInitializer, 
                            kernel_regularizer = l2(l2str)))
            brain.add(LeakyReLU(alpha = LeakyAlpha))
        brain.add(Dense(self.action_size, 
                        use_bias = True, 
                        kernel_initializer = KernelInitializer, 
                        bias_initializer = BiasInitializer, 
                        activation = "linear")) # output layer
        if GradClip.lower() == "norm":
            brain.compile(loss = "mse", 
                          optimizer = Adam(lr = self.learning_rate, 
                                           clipnorm = 1.0))
        elif GradClip.lower() == "value":             
            brain.compile(loss = "mse", 
                          optimizer = Adam(lr = self.learning_rate, 
                                           clipvalue = 1.0))
        else: # default: no clipping
            brain.compile(loss = "mse", 
                          optimizer = Adam(lr = self.learning_rate))
        # 列印模型摘要
        brain.summary(print_fn = Logger.info) # change print handler
        return brain
     
    def act(self, state, test=False):
        """ Acting policy of the DDQNAgent """
        act_values=[]
        action = np.zeros(self.action_size)
        if np.random.rand() <= self.epsilon and \
            self.train_test == "train" and not test:
            action[random.randrange(self.action_size)] = 1
        else:
            state = state.reshape(1, self.state_size)
            act_values = self.brain.predict(state)
            action[np.argmax(act_values[0])] = 1
        if test:
            return action, act_values
        else:
            return action
        
    def update_target_model(self):
        """
        Update weights for the DDQN agent, which is unnecessary for DQN agent
        """
        # 一段時間後，將兩組參數更新為相同
        self.brain_.set_weights(self.brain.get_weights())
        
    def observe(self, state, action, reward, next_state, done, \
                warming_up = False):
        """ Memory management and training of the agent """
        self.i = (self.i + 1) % self.memory_size
        self.memory[self.i] = (state, action, reward, next_state, done)
        if (self.i == self.memory_size - 1):
            # Logger.info("Memory Refilled.")
            pass
        if (not warming_up) and (self.i % self.train_interval) == 0:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decrement
                if self.epsilon < 0.:
                    self.epsilon = 0.
            state, action, reward, next_state, done = self._get_batches()
            # 報酬的計算公式與 DQN agent 不同
            # 這裡使用了兩組 DNN 模型參數 (brain & brain_)
            reward += (self.gamma
                       * np.logical_not(done)
                       * self.brain_.predict(next_state)\
                           [range(self.batch_size), \
                            (np.argmax(self.brain.predict(next_state), \
                                       axis = 1))])
            q_target = self.brain.predict(state)
            q_target[action[0], action[1]] = reward
            # 模型更新：DQN agent 無此步驟
            if self.i % self.update_target_freq == 0:
                self.update_target_model()
            return self.brain.fit(state, q_target,
                                  batch_size = self.batch_size,
                                  epochs = 1,
                                  verbose = False,
                                  validation_split = self.val_split)
        
    def _get_batches(self):
        """
        Selecting a batch of memory
        Split it into categorical subbatches
        Process action_batch into a position vector
        """
        batch = np.array(random.sample(self.memory, self.batch_size), \
                         dtype = object) # 2021.01.22 for ragged nested seq.
        state_batch = np.concatenate(batch[:, 0]) \
            .reshape(self.batch_size, self.state_size)
        action_batch = np.concatenate(batch[:, 1]) \
            .reshape(self.batch_size, self.action_size)
        reward_batch = batch[:, 2]
        next_state_batch = np.concatenate(batch[:, 3]) \
            .reshape(self.batch_size, self.state_size)
        done_batch = batch[:, 4]
        # action processing
        action_batch = np.where(action_batch == 1)
        return state_batch, action_batch, reward_batch, \
            next_state_batch, done_batch

class DDDQNAgent(Agent):
    """
    Dueling Double Deep Q-learning Network agent, 
    working learned agent, expected as the best agent choice.
    Separate Q* into two parts:
    Q*(s_t, a; theta, alpha, beta) = V(s_t; theta, beta) + 
    A(s_t, a; theta, alpha) - 1/|A|*sum(A(s_t, a'; theta, alpha))
    Where:
    V(s): value network with specific parameter beta
    A(s, a): advantage network with specific parameter alpha
    Referred to DQNAgent class for comments in details!
    """
    def __init__(self,
                 state_size,
                 action_size,
                 episodes,
                 episode_length, 
                 no_generators, 
                 train_interval = Train_Interval,
                 update_target_freq = Update_Target_Freq,
                 memory_size = Memory_Size,
                 gamma = Gamma,
                 learning_rate = Learning_Rate,
                 batch_size = Batch_Size,
                 epsilon_min = Epsilon_Min,
                 train_test = Train_Test,
                 symbol = Symbol, 
                 validation_split = Validation_Split):
        """ Initialize a DDDQN agent """
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.memory = [None] * memory_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decrement = (self.epsilon - epsilon_min) * \
            train_interval / (episodes * episode_length * no_generators)
        self.learning_rate = learning_rate
        self.train_interval = train_interval
        self.update_target_freq = update_target_freq
        self.batch_size = batch_size
        if Activation.lower() == "leakyrelu":
            self.brain = self._build_brain_2()
            self.brain_ = self._build_brain_2()
        else:
            self.brain = self._build_brain()
            self.brain_ = self._build_brain()
        self.i = 0
        self.train_test = train_test
        self.symbol = symbol
        self.val_split = validation_split
        
    def save_model(self):
        """ Save the current DDDQN agent model """
        fname = self.symbol + "_DDDQN.h5"
        Logger.info("Save model: %s" % fname)
        self.brain.save(Model_Path + '\\' + fname)
        
    def load_model(self):
        """ Load the current DDDQN agent model """
        fname = self.symbol + "_DDDQN.h5"
        Logger.info("Load model: %s" % fname)
        self.brain = load_model(Model_Path + '\\' + fname)
        
    def _build_brain(self):
        """ Build the agent's brain (deep neural network) """
        neurons_per_layer = Neurons_Per_Layer
        activation = Activation
        l2str = L2_Strength
        brain = Sequential()
        HLNPR = HiddenLayer_NParams_Ratio
        brain.add(Dense(neurons_per_layer * HLNPR[0], 
                        input_dim = self.state_size, 
                        use_bias = True, 
                        kernel_initializer = KernelInitializer, 
                        bias_initializer = BiasInitializer, 
                        activation = activation, 
                        kernel_regularizer = l2(l2str))) # input layer
        for i in range(1, len(HLNPR)):
            brain.add(Dense(neurons_per_layer * HLNPR[i], 
                            use_bias = True, 
                            kernel_initializer = KernelInitializer, 
                            bias_initializer = BiasInitializer, 
                            activation = activation, 
                            kernel_regularizer = l2(l2str)))
        brain.add(Dense(self.action_size, 
                        use_bias = True, 
                        kernel_initializer = KernelInitializer, 
                        bias_initializer = BiasInitializer, 
                        activation = "linear")) # original output layer
        # 以下與DQN和DDQN不同
        # 上面原本建立的 output layer 最後不會用到
        layer = brain.layers[-2]  # Get the second last layer of the model
        
        # 2021.04.14: use one of the following statements:
        nb_action = brain.output.shape[-1]
        # nb_action = brain.output.get_shape()[-1]
        # nb_action = brain.output_shape[-1]
        
        # 在 output layer 後方再加一層 y layer, size = actions + 1
        # (Value stream size = 1, advantage stream size = action_size)
        y = Dense(nb_action + 1, activation = "linear")(layer.output)
        
        # y layer (Value + Advantage) 後需要一層 output layer
        # Advantage stream 扣除均值，使 Q(s,a) = V(s) + A(s,a) 有唯一表示
        outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) \
                             + a[:, 1:] - K.mean(a[:, 1:], keepdims = True),
                             output_shape = (nb_action,))(y)
        # 指定 outputs 為新建立的
        brain = Model(inputs = brain.input, outputs = outputlayer)
        if GradClip.lower() == "norm":
            brain.compile(loss = "mse", 
                          optimizer = Adam(lr = self.learning_rate, 
                                           clipnorm = 1.0))
        elif GradClip.lower() == "value":             
            brain.compile(loss = "mse", 
                          optimizer = Adam(lr = self.learning_rate, 
                                           clipvalue = 1.0))
        else: # default: no clipping
            brain.compile(loss = "mse", 
                          optimizer = Adam(lr = self.learning_rate))
        # 列印模型摘要
        brain.summary(print_fn = Logger.info) # change print handler
        return brain
    
    def _build_brain_2(self):
        """ Build the agent's brain (deep neural network) """
        neurons_per_layer = Neurons_Per_Layer
        l2str = L2_Strength
        brain = Sequential()
        HLNPR = HiddenLayer_NParams_Ratio
        brain.add(Dense(neurons_per_layer * HLNPR[0], 
                        input_dim = self.state_size, 
                        use_bias = True, 
                        kernel_initializer = KernelInitializer, 
                        bias_initializer = BiasInitializer, 
                        kernel_regularizer = l2(l2str))) # input layer
        brain.add(LeakyReLU(alpha = LeakyAlpha))
        for i in range(1, len(HLNPR)):
            brain.add(Dense(neurons_per_layer * HLNPR[i], 
                            use_bias = True, 
                            kernel_initializer = KernelInitializer, 
                            bias_initializer = BiasInitializer, 
                            kernel_regularizer = l2(l2str)))
            brain.add(LeakyReLU(alpha = LeakyAlpha))
        brain.add(Dense(self.action_size, 
                        use_bias = True, 
                        kernel_initializer = KernelInitializer, 
                        bias_initializer = BiasInitializer, 
                        activation = "linear")) # original output layer
        # 以下與DQN和DDQN不同
        # 上面原本建立的 output layer 最後不會用到
        layer = brain.layers[-2]  # Get the second last layer of the model
        
        # 2021.04.14: use one of the following statements:
        nb_action = brain.output.shape[-1]
        # nb_action = brain.output.get_shape()[-1]
        # nb_action = brain.output_shape[-1]
        
        # 在 output layer 後方再加一層 y layer, size = actions + 1
        # (Value stream size = 1, advantage stream size = action_size)
        y = Dense(nb_action + 1, activation = "linear")(layer.output)
        
        # y layer (Value + Advantage) 後需要一層 output layer
        # Advantage stream 扣除均值，使 Q(s,a) = V(s) + A(s,a) 有唯一表示
        outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) \
                             + a[:, 1:] - K.mean(a[:, 1:], keepdims = True),
                             output_shape = (nb_action,))(y)
        # 指定 outputs 為新建立的
        brain = Model(inputs = brain.input, outputs = outputlayer)
        if GradClip.lower() == "norm":
            brain.compile(loss = "mse", 
                          optimizer = Adam(lr = self.learning_rate, 
                                           clipnorm = 1.0))
        elif GradClip.lower() == "value":             
            brain.compile(loss = "mse", 
                          optimizer = Adam(lr = self.learning_rate, 
                                           clipvalue = 1.0))
        else: # default: no clipping
            brain.compile(loss = "mse", 
                          optimizer = Adam(lr = self.learning_rate))
        # 列印模型摘要
        brain.summary(print_fn = Logger.info) # change print handler
        return brain

    def act(self, state, test=False):
        """
        Acting Policy of the DDDQNAgent
        """
        act_values = []
        action = np.zeros(self.action_size)
        if np.random.rand() <= self.epsilon and self.train_test == 'train' and not test:
            action[random.randrange(self.action_size)] = 1
        else:
            state = state.reshape(1, self.state_size)
            act_values = self.brain.predict(state)
            action[np.argmax(act_values[0])] = 1
        if test:
            return action, act_values
        else:
            return action
        
    def update_target_model(self):
        """
        Update weights for the DDQN agent, which is unnecessary for DQN agent
        """
        # 一段時間後，將兩組模型參數設為相同
        self.brain_.set_weights(self.brain.get_weights())
        
    def observe(self, state, action, reward, next_state, done, \
                warming_up = False):
        """
        Memory Management and training of the agent
        """
        self.i = (self.i + 1) % self.memory_size
        self.memory[self.i] = (state, action, reward, next_state, done)
        if (self.i == self.memory_size - 1):
            # Logger.info("Memory Refilled.")
            pass
        if (not warming_up) and (self.i % self.train_interval) == 0:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decrement
                if self.epsilon < 0.:
                    self.epsilon = 0.
            state, action, reward, next_state, done = self._get_batches()
            # 報酬的計算公式與 DQN 不同，和 DDQN 相同
            reward += (self.gamma
                       * np.logical_not(done)
                       * self.brain_.predict(next_state)\
                           [range(self.batch_size), \
                            (np.argmax(self.brain.predict(next_state), \
                                       axis = 1))])
            q_target = self.brain.predict(state)
            q_target[action[0], action[1]] = reward
            if self.i % self.update_target_freq == 0:
                self.update_target_model()
            return self.brain.fit(state, q_target,
                                  batch_size = self.batch_size,
                                  epochs = 1,
                                  verbose = False,
                                  validation_split = self.val_split)
        
    def _get_batches(self):
        """
        Selecting a batch of memory
        Split it into categorical subbatches
        Process action_batch into a position vector
        """
        batch = np.array(random.sample(self.memory, self.batch_size), \
                         dtype = object) # 2021.01.22 for ragged nested seq.
        state_batch = np.concatenate(batch[:, 0]) \
            .reshape(self.batch_size, self.state_size)
        action_batch = np.concatenate(batch[:, 1]) \
            .reshape(self.batch_size, self.action_size)
        reward_batch = batch[:, 2]
        next_state_batch = np.concatenate(batch[:, 3]) \
            .reshape(self.batch_size, self.state_size)
        done_batch = batch[:, 4]
        # action processing
        action_batch = np.where(action_batch == 1)
        return state_batch, action_batch, reward_batch, \
            next_state_batch, done_batch

# Done!
