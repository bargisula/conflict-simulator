# 🧠 Conflict Simulation Project

模擬國際衝突中的事件流程與模組觸發邏輯  
結合 NLP 預測、模組鏈式模擬引擎、可視化時間軸與自然語言敘述，幫助理解事件潛在影響與發展態勢。

---

## 📁 結構簡介

- `scripts/`：模型訓練、資料預處理、視覺化腳本
- `app/`：Streamlit 主程式（模擬 UI）
- `models/`：存放 `.pkl` 模型與向量器
- `data/`：原始與標註後的新聞資料
- `network/`：模組跳轉網路圖生成工具
- `notebooks/DEVLOG.md`：開發者日誌／筆記
- `.streamlit/`：Streamlit 佈景主題與設定檔

---

## 🧠 系統特色

- 🔍 **模組分類預測**：輸入新聞標題 → TF-IDF + 隨機森林模型推論可能模組
- 🔗 **模組觸發模擬引擎**：根據事件鏈式表預測模組間連動
- 📊 **互動式介面**：使用 Streamlit 呈現可操作的模擬介面
- 📋 **自然語言敘述解釋**：模擬結果自動生成逐模組文字說明
- ⏰ **視覺化時間軸**：模組觸發流程可視化呈現動態演進

---

## 🛠 安裝與執行方式

1. 安裝依賴：
   ```bash
   pip install -r requirements.txt

2.執行
   ```bash
   streamlit run app/conflict_simulator.py



製作中如有建議歡迎提出 🙌 由黃 X AI 合作設計模擬系統，持續進化中 🧠✨
