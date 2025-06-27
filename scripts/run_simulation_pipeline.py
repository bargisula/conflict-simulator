import pandas as pd
import joblib
import os
import random

# 模擬引擎參數（可調整）
TRIGGER_CHAIN = {
    "R2": ["C1"],
    "C1": ["R4", "C2"],
    "R4": ["R3"],
    "R3": [],
    "C2": []
}

MODULE_DELAY = {
    "R2": 1,
    "C1": 1,
    "R4": 2,
    "R3": 2,
    "C2": 1
}

FOLLOW_TRIGGER_PROB = 0.85  # 模組被觸發後真的發生的機率
SIM_THRESHOLD = 0.4  # 機率超過才算觸發第一模組

# 載入模型
def load_model():
    tfidf = joblib.load("../models/tfidf.pkl")
    model = joblib.load("../models/model.pkl")
    le = joblib.load("../models/label_encoder.pkl")
    return tfidf, model, le

# 模組預測
def predict_modules(title, tfidf, model, le):
    X = tfidf.transform([title])
    probas = model.predict_proba(X)[0]
    result = [
        {"module": le.classes_[i], "prob": round(p, 4)}
        for i, p in enumerate(probas) if p >= SIM_THRESHOLD
    ]
    return sorted(result, key=lambda x: -x["prob"])

# 模擬引擎：模組觸發
def simulate_round_trace(initial_modules):
    triggered_log = []
    visited = set()
    t = 0

    queue = [(mod["module"], None, mod["prob"]) for mod in initial_modules]

    while queue:
        current, triggered_by, prob = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        triggered_log.append({
            "module": current,
            "triggered_by": triggered_by or "Model",
            "probability": round(prob, 4),
            "time": t
        })
        for next_mod in TRIGGER_CHAIN.get(current, []):
            if random.random() <= FOLLOW_TRIGGER_PROB:
                delay = MODULE_DELAY.get(next_mod, 1)
                queue.append((next_mod, current, prob * FOLLOW_TRIGGER_PROB))
        t += 1

    return pd.DataFrame(triggered_log)

# 主流程：模組預測 + 模擬
def run(title):
    tfidf, model, le = load_model()
    print(f"\n📰 新聞標題：{title}")
    preds = predict_modules(title, tfidf, model, le)
    print("🎯 預測模組：", [f"{p['module']} ({p['prob']})" for p in preds])
    df_sim = simulate_round_trace(preds)
    print("\n📋 模組觸發流程：\n", df_sim)
    return df_sim

if __name__ == "__main__":
    # 📝 測試新聞（可換成你的案例）
    sample_title = "Iran launched missiles toward Israel amid rising tensions"
    df = run(sample_title)

    # ✅ 儲存模擬歷程（可關掉）
    os.makedirs("../data", exist_ok=True)
    df.to_csv("../data/simulation_log.csv", index=False)
    print("\n✅ 模擬結果已儲存至 data/simulation_log.csv")
