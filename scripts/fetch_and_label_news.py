import requests
import pandas as pd
from dotenv import load_dotenv
import os

# 載入 .env 的環境變數
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

# 模組關鍵字對照
keyword_module_map = {
    "R2": ["伊朗", "Iran", "飛彈", "missile", "導彈", "襲擊"],
    "C1": ["以色列", "Israel", "反擊", "strike back", "空襲"],
    "R4": ["真主黨", "Hezbollah", "proxy", "黎巴嫩", "武裝組織"],
    "R3": ["荷莫茲", "Hormuz", "海峽", "油輪", "封鎖", "naval"],
    "C2": ["停火", "ceasefire", "和談", "peace talks", "外交", "調停"]
}

# 模組標註函式（允許多個模組）
def label_modules(title):
    matched = set()
    title = str(title).lower()
    for mod, keywords in keyword_module_map.items():
        for kw in keywords:
            if kw.lower() in title:
                matched.add(mod)
    return list(matched) if matched else ["UNK"]

# 抓新聞 & 自動標註流程
def fetch_and_label_news(query="Iran AND missile OR Israel", page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": API_KEY
    }

    res = requests.get(url, params=params)
    if res.status_code != 200:
        print(f"❌ 抓取失敗：{res.status_code} - {res.text}")
        return

    articles = res.json().get("articles", [])
    records = []
    for a in articles:
        title = a["title"]
        modules = label_modules(title)
        records.append({
            "title": title,
            "description": a["description"],
            "publishedAt": a["publishedAt"],
            "source": a["source"]["name"],
            "url": a["url"],
            "modules": modules
        })

    df = pd.DataFrame(records)
    df.to_csv("../data/newsapi_labeled.csv", index=False)
    print(f"✅ 已抓取並標註 {len(df)} 筆新聞 → 儲存至 data/newsapi_labeled.csv")

if __name__ == "__main__":
    fetch_and_label_news()
