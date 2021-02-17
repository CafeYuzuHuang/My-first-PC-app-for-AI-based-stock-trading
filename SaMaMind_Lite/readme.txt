# 關於SaMaMind Lite: (目前版本別：0.0)

SaMaMind名字取自artificial intelligence MIND for SalaryMAn, 顧名思義為針對沒時間做證券技術分析的上班族散戶們提供一個盤後操作工具，作為日後上市櫃證券(股票與ETF)掛單交易的參考。

SaMaMind唸起來則像是Shaman Mind(薩滿的才智)而增添了一點神祕感。但說破了，它背後運行的原理其實為深度增強式學習法(最著名的代表為Alphago)。SaMaMind藉由事先準備好的資料庫與使用者定義之超參數設定進行模型訓練，訓練目標為在指定的時間範圍區間內取得最大報酬(或最佳績效表現，注意兩者意義不同)。

SaMaMind Lite為SaMaMind的簡化版本。



# 改版歷程：

- 2021.02.17：正式發佈版本，版號0.0。
- 2021.02.15：內部測試Alpha版本。



# 軟體需求：

- Windows作業系統
- Windows版本之網路瀏覽器，支援無頭模式(e.g. Chrome or PhantomJS driver)
- Python 3.7.7
- Tensor-flow 虛擬環境
	- tensorflow 2.3.0
	- keras 2.4.3
- 其餘python模組，部分需額外透過pip或conda install:
	- arch
	- beautifulsoup4
	- matplotlib
	- mplfinance
	- numpy
	- pandas
	- pillow (PIL)
	- re
	- requests
	- scipy
	- seaborn
	- selenium
	- sklearn
	- statsmodels
	- tk (需使用Ttk模組，版本要求為8.5以上)
- 內建python模組
	- collections
	- datetime
	- logging
	- math
	- os
	- random
	- shutil
	- time
	- threading
	- webbrowser



# 操作使用與建議：

- 執行方式：
	(1) 使用IDE如Spyder，於tensorflow環境下執行SaMaMind_Lite.py腳本
	(2) 使用如Anaconda prompt，依序執行以下指令：
		activate [你的tensorflow虛擬環境名稱]
		cd [你存放SaMaMind_Lite.py的資料夾路徑]
		python SaMaMind_Lite.py (不需任何其他引數與選項設定)
	注意(1)與(2)的顯示圖片格式可能不同，但儲存之圖檔應相同
- 建立爬蟲清單與超參數清單，存放於”user infiles”資料夾中各自的子資料夾內；
- 爬蟲下載的資料會放在”data scraped”資料夾中，後處理過後會放在”data postprocessed”資料夾中；若從證交所下載資料，則會將新資料疊加到舊資料後面(重複資訊刪除)；
- “data ai-model”存放.h5格式之模型權重參數，”data ai-trade”存放DRL運算結果(需定期刪除)；
- “log”資料夾內存放眾多日誌檔，需定期手動刪除
- 爬蟲所使用的url與headers資訊需定期更新，網站格式如有改變，爬蟲程式碼也需要定期對應變更
- 部分工具並未建立於GUI上，可透過腳本與ipython操作作調用。



# 未來改版方向：

- 將資料庫與MySQL結合操作；
- 全球股市資訊擷取 (e.g. NYSE, NASDAQ, …)；
- 更好用與高階的使用者圖形介面；
- 更有彈性的特徵設定(DRL環境可觀察資訊)，例如允許觀測連續數日的K棒資訊(就像是Alphago Zero可以觀測前幾手的著手順序一樣)；
- 更有彈性的模型訓練量度，例如以績效指標如夏普值取代mse或同時考慮持倉週期與交易覆蓋率(持倉總天數除以交易總天數)；
- 加速DRL模型計算，使用CPU平行運算或GPU，或僅透過演算法改良。
- 自動建立爬蟲標的表單，例如取得當下的台灣五十成分股。



# 授權資訊：無。



# 免責聲明：

本程式僅供技術與學術交流之用，對於使用者以下行為，作者不擔負任何責任：
(1) 以其取得與揭露之市場資訊正確性失當，所造就之個人損失；
(2) 因其市場資訊落後於現況，其所造就之個人損失；
(3) 遵循程式進行交易之個人損失。在此重申本程式所建議之交易建議與績效評估僅供技術與學術交流之用，不構成任何標的之買賣邀約；
(4) 任何未列於上，但逾越技術與學術交流之行為，例如任何形式的商業行為(程式績效宣傳、程式買賣與授權等)。



# 更多資訊…

- 拜訪作者：https://github.com/CafeYuzuHuang
- DRL腳本參考：https://github.com/saeed349/Deep-Reinforcement-Learning-in-Trading
SaMaMind Lite所使用的DRL腳本修改自Saeed349之程式碼以符合需求。



# ***   *** #



# About SaMaMind Lite: (Current ver.: 0.0)

SaMaMind stands for artificial intelligence MIND for SAlaryMAn, which sounds like “Shaman” Mind in coincidence. It aims for Taiwan securities (TSE and OTC stocks and ETFs) price technical analysis and trading action suggestion via deep reinforcement learning (DRL; Alphago is the well-known one worldwide). SaMaMind’s DRL model is trained by prepared stock price database with hyper-parameters set by users, aiming for trading with the optimum rewards (or best performance) within specified time span.

SaMaMind Lite is the simplified version of SaMaMind.



# Version log:

- 2021.02.17: version 0.0; first released.
- 2021.02.15：developer's version alpha.



# Requirements:

- Windows OS
- Webdriver for Windows OS (e.g. Chrome or PhantomJS headless driver)
- Python 3.7.7
- Tensor-flow virtual environment
	- tensorflow 2.3.0
	- keras 2.4.3
- Other python modules listed alphabetically:
	- arch
	- beautifulsoup4
	- matplotlib
	- mplfinance
	- numpy
	- pandas
	- pillow (PIL)
	- re
	- requests
	- scipy
	- seaborn
	- selenium
	- sklearn
	- statsmodels
	- tk (with Ttk, 8.5 or later)
- Build-in python modules listed alphabetically:
	- collections
	- datetime
	- logging
	- math
	- os
	- random
	- shutil
	- time
	- threading
	- webbrowser



# Usage:

- How to use SaMaMind Lite?
	(1) Use IDE such as Spyder, with tensorflow virtual environment, execute the script SaMaMind_Lite.py.
	(2) Use Anaconda prompt, enter the following commands:
		activate [your tensorflow virtual environment name]
		cd [the folder path where contains SaMaMind_Lite.py]
		python SaMaMind_Lite.py (no other arguments and option settings necessary!)
	Notice that for (1) and (2) the displayed format of GUI and graphs may differ, but the exported graphs should be the same!
- The user-defined scraping list .in files and hyperparameter set .in files are saved in parent folder ”user infiles”; 
- The on-line scraped data will be saved in folder named ”data scraped”; what are post-processed will be ”data postprocessed”. Note that if the data is scraped from the official site, the data will be appended to the older existing files with removing duplicate rows;  
- The keras H5 formatted DNN weight data is restored in “data ai-model” folder; “data ai-trade” restores the outputs of DRL computing, which should be cleared routinely;
- Log files are saved in folder “log”, which should be cleared manually and routinely;
- Url and headers should be routinely updated, so does the corresponding scripting, too;
- Some functionalities are hided from the GUI, which should be applied via Ipython interface.



# What’s next:

- Database operation combined with MySQL.
- World-wide data scraping (e.g. NYSE, NASDAQ, …)
- More user-friendly and higher-level GUI …
- More flexible feature selection (environment observation).
- Options for DRL training metrics, e.g. replace mse by Sharpe ratio or other performance indicators; considering average hold periods or cover rate (total hold days divided by total tradable days).
- Accelerating techniques for DRL computing; e.g. multi-core CPUs or GPU supported, or simply algorithm improvement.
- Automated listing for data scraping, such as TW50 stocks.



# Licensing: None.



# Disclaimer: 

SaMaMind Lite aims for academic purpose only! The author has no responsiblity for the accuracy and instantaneity of data collected and represented, as well as the financial loss and commercial behaviors by users who follows the information provided by SaMaMind Lite.



# For more information:

- Author's website: https://github.com/CafeYuzuHuang
- Source of DRL script: https://github.com/saeed349/Deep-Reinforcement-Learning-in-Trading


