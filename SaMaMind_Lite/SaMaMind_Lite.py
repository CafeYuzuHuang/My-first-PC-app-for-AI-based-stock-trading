# -*- coding: utf-8 -*-
"""
< 2021.02.17 > Project structure
Modules and scripts:
 └ ─ ─ SaMaMind_Lite.py
 └ ─ ─ myStockScraper
        └ ─ ─ ─ __init__.py
        └ ─ ─ ─ myScraperSettings.py
        └ ─ ─ ─ myScraper.py
        └ ─ ─ ─ myPostProcessing.py
 └ ─ ─ myMind
        └ ─ ─ ─ __init__.py
        └ ─ ─ ─ myWorldSettings.py
        └ ─ ─ ─ myWorld.py
        └ ─ ─ ─ myEnv.py
        └ ─ ─ ─ myAgent.py
        └ ─ ─ ─ myAna.py
        └ ─ ─ ─ myDNNVisualizer.py
 └ ─ ─ myGUI
        └ ─ ─ ─ __init__.py
        └ ─ ─ ─ myApp.py
 └ ─ ─ myUtilities
        └ ─ ─ ─ __init__.py
        └ ─ ─ ─ myLog.py
        └ ─ ─ ─ myParams.py
        └ ─ ─ ─ myTechIndices.py
        └ ─ ─ ─ myTSTests.py
 └ ─ ─ myTest
        └ ─ ─ ─ __init__.py
        └ ─ ─ ─ myScraperTest.py
        └ ─ ─ ─ myRLTest.py
        └ ─ ─ ─ myGPUTimeTest.py

Directories for data storage:
 └ ─ ─ log
 └ ─ ─ user infiles
        └ ─ ─ ─ user pool
        └ ─ ─ ─ ai param set
 └ ─ ─ images
 └ ─ ─ webdrivers
 └ ─ ─ data scraped
 └ ─ ─ data postprocessed
 └ ─ ─ data ai-model
 └ ─ ─ data ai-trade
        └ ─ ─ ─ anime
"""

from os import environ
from time import sleep
# from myGUI import myApp

"""
SaMaMind Lite之主程式，用於 GUI 主視窗操作與參數讀取
可透過Spyder等IDE介面或Anaconda prompt之命令列操作
"""

def AllocateGPUDevice(val = 0):
    """ GPU裝置設定與訊息輸出 """
    try:
        # default: '0'; assign '-1' for CPU only
        environ['CUDA_VISIBLE_DEVICES'] = str(val)
        if int(environ['CUDA_VISIBLE_DEVICES']) < 0:
            print("SaMaMind_Lite applies CPU only for DRL computing ...")
        else:
            print("SaMaMind_Lite attempts to use GPU for DRL computing.")
            print("Use GPU device: ", environ['CUDA_VISIBLE_DEVICES'])
            print("(Default: '0')")
    except Exception as e:
        print(e)
        print("Please check env variable setting!")
    return None


### SaMaMind_Lite launch code ###
# AllocateGPUDevice(-1) # CPU only
AllocateGPUDevice() # default GPU device
sleep(1)

from myGUI import myApp
sleep(1)

window = myApp.MainWindow()
# window.mainloop(True) # Fixed window size
window.mainloop() # Default: resizable window
# window.quitloop()
sleep(1)
# Done!!

