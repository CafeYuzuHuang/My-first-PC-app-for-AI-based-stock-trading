# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# Require pydot (and pydotplus) module(s) installed within the 
# tensorflow environment.
from keras.utils.vis_utils import plot_model
from keras.models import load_model
# from keras import backend as K

from myUtilities import myParams

"""
myDNNVisualizer.py的功能：
    (1) 將 DNN 參數視覺化
    (2) 匯出 DNN 架構描述
"""

plt.set_loglevel("warning") # avoid log polluted by mpl

def VisualizeNetwork(k_layer, fpath = None):
    """
    Visualization for single network weights
    k_layer should be numpy.ndarray
    """
    # matplotlib default settings:
    mpl.style.use("classic")
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    fig, (ax, cbar_ax) = plt.subplots(2, figsize = (12, 8), \
                                      gridspec_kw = grid_kws)
    ax = sns.heatmap(k_layer, cmap = cm.coolwarm, ax = ax, \
                     cbar_ax = cbar_ax, \
                     cbar_kws = {"orientation": "horizontal"})
    plt.show()
    if not fpath is None:
        plt.savefig(fpath)
    return None

def VisualizeDNN(k_weights, in_name, out_name, fpath = None):
    """
    Deep neural network visualization for keras model.
    Some statements are modified (r.f. whose comment with keyword "keras")
    Info displayed:
        (1) Name of input(s) and output(s) (text)
        (2) # of neurons in each layer (text)
        (3) Coefficients (weights) (colored lines)
        (4) Biases of layers are not shown in the current version
    """
    # matplotlib default settings:
    mpl.style.use("classic")
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    # Initialization
    neurons_each_layer = []
    neurons_coor_x = []
    neurons_coor_y = []
    in_txt_coor_x = []
    in_txt_coor_y = []
    out_txt_coor_x = []
    out_txt_coor_y = []
    hid_txt_coor_x = []
    hid_txt_coor_y = []
    
    # Get info
    # sklearn version
    """
    n_connects_layer = len(dnn_coef)
    for ii in range(0, n_connects_layer):
        # Get # of inputs and neurons of each hidden layer
        neurons_each_layer.append(len(dnn_coef[ii]))
    neurons_each_layer.append(len(dnn_coef[-1][0])) # # of outputs
    max_neurons = max(neurons_each_layer)
    # min_neurons = min(neurons_each_layer)
    """
    
    # keras version
    # Generate dnn_coef from k_weights
    dnn_coef = []
    for i in range(len(k_weights)):
        if len(k_weights[i].shape) == 2: # network
            dnn_coef.append(k_weights[i])
        else: # bias (and undefined)
            pass
    print("Check extracted results: ")
    for j in range(len(dnn_coef)):
        print("Layer: %d, shape: (%d, %d)." % \
              (j+1, dnn_coef[j].shape[0], dnn_coef[j].shape[1]))
    
    n_connects_layer = len(dnn_coef)
    for ii in range(n_connects_layer):
        # Get # of inputs and neurons of each hidden layer
        neurons_each_layer.append(dnn_coef[ii].shape[0])
    # No. of outputs
    # For DDDQN, it is no. of outputs of lambda layer = real no. outputs + 1
    neurons_each_layer.append(dnn_coef[-1].shape[1])
    max_neurons = max(neurons_each_layer)
    
    # Assign draw parameters
    dx_layers = 100.0 # distance between layers
    dy_neurons = 0.6*dx_layers # distance between neurons
    dx_space = 4.0*dx_layers # spacing # keras
    dy_space = 4.0*dx_layers # spacing # keras
    txt_space = 1.2*dx_layers # spacing # keras
    r_neuron = 0.25*dx_layers # radius of each neuron
    dx_layers = round(r_neuron*max_neurons, -2) # updated
    x_max = dx_layers*n_connects_layer + dx_space*2.0
    y_max = dy_neurons*(max_neurons - 1.0) + dy_space*2.0
    Lwidth = 1 # connection line width
    
    # Assign coordinates of each neuron
    x_layer_cur = dx_space # current x position
    y_mid = 0.5*y_max # midpoint of y position
    for ii in range(0, len(neurons_each_layer)):
        n = neurons_each_layer[ii]
        x_coor = []
        y_coor = []
        tmp_y = []
        if n % 2 == 1: # odd value
            # e.g. if 7 neurons -> 3, 2, 1, 0, -1, -2, -3
            y_start = n // 2
            y_end = -1*(n // 2) - 1
            tmp_y = np.arange(y_start, y_end, -1)
        else: # even value
            # e.g. if 8 neurons -> 3.5, 2.5, 1.5, 0.5, -0.5, -1.5, -2.5, -3.5
            y_start = (n // 2) - 0.5
            y_end = -1*(n // 2) - 0.5
            tmp_y = np.arange(y_start, y_end, -1)   
        for jj in range(0, len(tmp_y)):
            x_coor.append(x_layer_cur)
            y_cur = y_mid + float(tmp_y[jj])*dy_neurons
            y_coor.append(y_cur)
        neurons_coor_x.append(x_coor) # collect x coordinates of neurons
        neurons_coor_y.append(y_coor) # collect y coordinates of neurons
        
        if ii == 0: # input layer
            in_txt_coor_x = [x_layer_cur - txt_space]*neurons_each_layer[ii]
            # in_txt_coor_y = y_coor
            in_txt_coor_y = [(2*kk - y_mid) for kk in y_coor] # keras
        elif ii == n_connects_layer: # output layer
            out_txt_coor_x = [x_layer_cur + txt_space]*neurons_each_layer[ii]
            # out_txt_coor_y = y_coor
            out_txt_coor_y = [(2*kk - y_mid) for kk in y_coor] # keras
        else: # hidden layers
            hid_txt_coor_x.append(x_layer_cur)
            hid_txt_coor_y.append(max(y_coor) + txt_space)
        x_layer_cur += dx_layers # go to next layer
    
    # Start plotting:
    # plt.figure() # default figsize = 6.4 x 4.8 (inches)
    # plt.figure(figsize = (8, 6))
    plt.figure(figsize = (16, 12)) # keras
    ax = plt.gca()
    # Deal with neurons
    for ii in range(0, len(neurons_each_layer)):
        for jj in range(0, neurons_each_layer[ii]):
            x = neurons_coor_x[ii][jj]
            y = neurons_coor_y[ii][jj]
            DNN_Nodes = plt.Circle((x, y), r_neuron, ec = 'k', fc = 'w', \
                                   zorder = 4) # higher zorder draws later
            ax.add_artist(DNN_Nodes)
    
    # Deal with links
    # Get the RGBA values from a float, the shape of list is i x j x k x 4
    colors = [cm.coolwarm(color) for color in dnn_coef]
    dnn_flatten = [] # Flatten dnn_coef: used later
    for ii in range(0, n_connects_layer):
        # Connection network between layer ii and ii+1:
        for jj in range(0, neurons_each_layer[ii]):
            for kk in range(0, neurons_each_layer[ii+1]):
                xj = neurons_coor_x[ii][jj]
                yj = neurons_coor_y[ii][jj]
                xk = neurons_coor_x[ii+1][kk]
                yk = neurons_coor_y[ii+1][kk]
                # wijk = dnn_coef[ii][jj][kk] # weight of the connection
                DNN_Edges = plt.Line2D([xj, xk], [yj, yk], linewidth = Lwidth, \
                                       c = colors[ii][jj][kk], zorder = 1)
                ax.add_artist(DNN_Edges)
                dnn_flatten.append(dnn_coef[ii][jj][kk])
    
    # Deal with labels
    for ii in range(0, len(neurons_each_layer)):
        if ii == 0: # input layer
            for jj in range(0, neurons_each_layer[ii]):
                plt.text(in_txt_coor_x[jj], in_txt_coor_y[jj], in_name[jj], \
                         color = 'g', ha = 'right', va = 'center')           
        elif ii == n_connects_layer: # output layer
            for jj in range(0, neurons_each_layer[ii]):
                plt.text(out_txt_coor_x[jj], out_txt_coor_y[jj], out_name[jj], \
                         color = 'g', ha = 'left', va = 'center')           
        else: # hidden layers
            plt.text(hid_txt_coor_x[ii-1], max(hid_txt_coor_y), \
                     str(neurons_each_layer[ii]), color = 'g', \
                     ha = 'center', va = 'center')
    
    # Use scatter plot:
    # The datapoints are hidden by set marker size = 0, thus (x, y) can be
    # arbitrarily assigned (shape of x and y should be i x j x k)
    plt.xlim([0, x_max])
    plt.ylim([0, y_max])
    ax.set_aspect(1.0)
    plt.scatter(dnn_flatten, dnn_flatten, s = 0, c = dnn_flatten, \
                cmap = 'coolwarm')
    plt.colorbar() # Colorbar is the thing what we need here
    plt.axis('off')
    plt.show() # Done!
    if not fpath is None:
        plt.savefig(fpath)
    return None

def Load_Model(m_path, **kwargs):
    """
    載入keras .h5格式模型
    為 Keras 中 load_model 的 wrapper，參數定義與之相同
    """
    return load_model(m_path, **kwargs)

def Export_Model_Struct(k_model, **kwargs):
    """
    輸出關於模型的架構描述
    為 Keras 中 plot_model 的 wrapper，參數定義與之相同
    """
    plot_model(k_model, **kwargs)
    return None

def Print_Model_Struct(k_model):
    """ 螢幕輸出模型的架構描述 """
    weights = k_model.get_weights()
    n = len(weights)
    go_down = True
    i = 0
    nl = 0
    print("==============  i = 1  ===============")
    while go_down:
        if len(weights[i].shape) == 2:
            nl += 1
            print("Layer: %d, network weight shape: (%d, %d) " % \
                  (nl, weights[i].shape[0], weights[i].shape[1]))
        elif len(weights[i].shape) == 1:
            print("Layer: %d, bias shape: (%d, ) " % \
                  (nl, weights[i].shape[0]))
        else:
            print("Layer: %d, undefined, shape: ", weights[i].shape)
        i += 1
        if i >= n:
            go_down = False # exit
            print("====================================")
        else:
            print("==============  i = %d  ===============" % (i+1))
    print("Total # of dense layers: %d" % nl)
    if i > nl:
        print("Bias is applied to these dense layers.")
    print("Caution: Lambda layer will not be presented.")
    print("(Use summary method of the keras model instead.)")
    print("====================================")
    return None

# Testing
if __name__ == '__main__':
    # Keras .h5 格式權重參數儲存路徑
    Model_Path = os.path.dirname(os.getcwd()) + \
        '\\' + myParams.AI_Foldername
    if not os.path.isdir(Model_Path):
        os.mkdir(Model_Path)
    model_name = "demo_test_DDDQN.h5"
    fpath = Model_Path + '\\' + model_name
    pydot_path = Model_Path + '\\' + "demo_test_DDDQN.png"
    
    # 檢查讀取的檔案
    dnn = load_model(fpath)
    print("Summary of the keras model")
    print("file path: ", fpath)
    dnn.summary() # Show the model summary
    Print_Model_Struct(dnn)
    # 輸出模型的架構描述
    # plot_model(dnn, to_file = pydot_path, show_shapes = True, \
    #            show_layer_names = True)
    Export_Model_Struct(dnn, to_file = pydot_path, \
                        show_shapes = True, \
                        show_layer_names = True) # wrapper of plot_model
    
    # 圖像化
    vis_path = Model_Path + '\\' + "demo_test_DDDQN_network.png"
    vis_path_1 = Model_Path + '\\' + "demo_test_DDDQN_heatmap.png"
    ins = dnn.get_weights()[0].shape[0]
    outs = dnn.get_weights()[-1].shape[0]
    innames = ["In-" + str(i+1) for i in range(ins)]
    outnames = ["Out-" + str(i+1) for i in range(outs)]
    # get_weights() 方法回傳一個 list，成員為 numpy.ndarray 類型
    # 其中成員可能為1-d或2-d array，須以 shape 屬性確認
    dnncoef = dnn.get_weights()
    # VisualizeNetwork(dnncoef[0], vis_path_1)
    VisualizeNetwork(dnncoef[4], vis_path_1)
    # VisualizeNetwork(dnncoef[-1], vis_path_1) # cause error if input bias
    VisualizeDNN(dnncoef, innames, outnames, vis_path)
# Done!
