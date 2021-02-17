# -*- coding: utf-8 -*-

from collections import namedtuple

from myStockScraper.myPostProcessing import TradePeriod

"""
RL 模型參數組，用於 myWorld.py
"""

# Define data structure for the parameter set
# 7 parameters in myWorld
# 7 parameters in myEnv
# 13 parameters in myAgent
ParamNames = ["trading_tax", 
              "trading_fee", 
              "time_fee", 
              "train_test_split", 
              "episodes", 
              "episode_length", 
              "history_length", 
              "prep_isreturnbasis", 
              "initial_cash", 
              "max_underwater_ratio", 
              "keep_cash_ratio", 
              "spread",
              "unit_quantity", 
              "hold_period_upper", 
              "chosen_agent", 
              "gamma", 
              "epsilon_min", 
              "neurons_per_layer", 
              "hiddenlayer_shape", 
              "activation", 
              "l2_strength", 
              "learning_rate", 
              "memory_size", 
              "batch_size", 
              "train_interval", 
              "update_target_freq", 
              "validation_split"]
RLParamSet = namedtuple("RLParams", ParamNames)

# Default parameter set for debugging and model testing
RL_Test_Set = RLParamSet(0.003, 
                         0.001425, 
                         0.01/365, 
                         0.8, 
                         10, 
                         300, 
                         2, 
                         True, 
                         1000000, 
                         -0.5, 
                         0.5, 
                         0.005, 
                         1000, 
                         TradePeriod, 
                         "DDDQN", 
                         0.95, 
                         0.01, 
                         24, 
                         [1, 2, 4], 
                         "relu", 
                         0.01, 
                         0.001, 
                         1024, 
                         64, 
                         100, 
                         100, 
                         0.2)

# Store all parameter sets to dict
RL_Set_Dict = {"default_test": RL_Test_Set}

# Done!
