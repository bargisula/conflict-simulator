import requests
import pandas as pd
from dotenv import load_dotenv
import os

# ✅ 載入環境變數
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")
QUERY = "Iran AND missile OR Israel"
LANGUAGE = "en"
PAGE_SIZE = 50
OUTPUT_CSV = "../data/newsapi_iran_missile.csv"

def fetch_articles():
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": QUERY,
        "language": LANGUAGE,
        "pageSize": PAGE_SIZE,
        "sortBy": "publishedAt",
        "apiKey": API_KEY
    }

    res = requests.get(url, params=params)
    if res.status_code != 200:
        print(f"❌ 取得資料失敗：{res.status_code}")
        print(res.text)
        return

    articles = res.json().get("articles", [])
    data = [{
        "title": a["title"],
        "description": a["description"],
        "publishedAt": a["publishedAt"],
        "source": a["source"]["name"],
        "url": a["url"]
    } for a in articles]

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 已儲存新聞 {len(df)} 筆至 {OUTPUT_CSV}")

if __name__ == "__main__":
    fetch_articles()
