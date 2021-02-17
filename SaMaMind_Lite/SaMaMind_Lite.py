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

from time import sleep
from myGUI import myApp

"""
SaMaMind Lite之主程式，用於 GUI 主視窗操作與參數讀取
可透過Spyder等IDE介面或Anaconda prompt之命令列操作
"""

if __name__ == "__main__":
    sleep(1)
    window = myApp.MainWindow()
    window.mainloop() # Default: resizable window
    # window.quitloop()
    sleep(1)
# Done!
