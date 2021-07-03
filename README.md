# My-first-PC-app-for-AI-based-stock-trading
A simple graphical application for web scraping, AI training and testing, trading performance assessment, and so on.

---
## 寫於2021.07.04:

這是一套透過深度強化學習(DRL)來進行台股交易分析的小型應用程式，其[Lite版本](https://github.com/CafeYuzuHuang/My-first-PC-app-for-AI-based-stock-trading/tree/main/SaMaMind_Lite)包含以下功能：
* 網路擷取台股歷史股價資料(證交所、雅虎財經)
* DRL模型的設定、訓練，與測試
* 交易點位顯示、資產水位圖
* 績效評估
* 帶技術指標的K線圖繪製

GUI外觀樣式與簡單操作如[示範影片](https://www.youtube.com/watch?v=NAWkuPWJoBg&fbclid=IwAR0WScgTwqKfsZrd5ybGzkyJLofXjuG5pfoRGVtRBzeplJZD7TV0oMfgtY0)所示

### 功能補充說明：
* 部分功能例如時間序列分析並未寫入GUI
* 個股的回測結果檢視除了透過GUI功能，也可以使用這個[小工具](https://github.com/CafeYuzuHuang/SaMaMind_Lite-web-plot-tool)來達成
* 隔日交易建議與其他更高階的功能目前不在SaMaMind的Lite版本中，或許哪一天會放上來

### Demo:
DRL模型視覺化與分析結果出圖可參考[範例檔](https://github.com/CafeYuzuHuang/My-first-PC-app-for-AI-based-stock-trading/tree/main/Example%20data)。在這個範例中，模型訓練結果顯示了一個[經典的課題](https://towardsdatascience.com/the-dying-relu-problem-clearly-explained-42d0c54e0d24)，儘管其績效測試結果尚可。

[近期較新的範例檔](https://github.com/CafeYuzuHuang/My-first-PC-app-for-AI-based-stock-trading/tree/main/Example%20data%2020210528)中選用台灣上市前50大權值股作為標的，將DRL模型與買進持有策略進行比較。結果顯示該模型參數優於買進持有策略，原因在於擇時進場避開不必要的風險，與適時的反向做空。

---

