import streamlit as st
import pandas as pd
import joblib
import random
import altair as alt

# 模型載入
@st.cache_resource
def load_model():
    tfidf = joblib.load("../models/tfidf.pkl")
    model = joblib.load("../models/model.pkl")
    le = joblib.load("../models/label_encoder.pkl")
    return tfidf, model, le

# 預測模組
def predict_modules(title, tfidf, model, le, threshold=0.4):
    X = tfidf.transform([title])
    probas = model.predict_proba(X)[0]
    return sorted([
        {"module": le.classes_[i], "prob": round(p, 4)}
        for i, p in enumerate(probas) if p >= threshold
    ], key=lambda x: -x["prob"])

# 模擬流程
TRIGGER_CHAIN = {
    "R2": ["C1"],
    "C1": ["R4", "C2"],
    "R4": ["R3"],
    "R3": [],
    "C2": []
}

MODULE_DELAY = {"R2":1,"C1":1,"R4":2,"R3":2,"C2":1}
FOLLOW_TRIGGER_PROB = 0.85

def simulate_round_trace(initial_modules):
    log, visited, t = [], set(), 0
    queue = [(m["module"], None, m["prob"]) for m in initial_modules]
    while queue:
        curr, src, prob = queue.pop(0)
        if curr in visited:
            continue
        visited.add(curr)
        log.append({
            "module": curr,
            "triggered_by": src or "Model",
            "probability": round(prob, 4),
            "time": t
        })
        for next_m in TRIGGER_CHAIN.get(curr, []):
            if random.random() <= FOLLOW_TRIGGER_PROB:
                delay = MODULE_DELAY.get(next_m, 1)
                queue.append((next_m, curr, prob * FOLLOW_TRIGGER_PROB))
        t += 1
    return pd.DataFrame(log)

# 模組時間軸圖
def plot_timeline(df):
    return alt.Chart(df).mark_bar(size=20).encode(
        y=alt.Y('module:N', sort=alt.EncodingSortField("time", order="ascending"), title="模組"),
        x=alt.X('time:Q', title="觸發時間"),
        color=alt.Color('triggered_by:N', title="來源"),
        tooltip=['module', 'triggered_by', 'probability', 'time']
    ).properties(title="模組觸發時間軸")

# UI
st.set_page_config(page_title="Conflict Simulation", page_icon="🧠", layout="wide")
st.title("🧠 Conflict Simulation 模擬器")

user_input = st.text_input("📰 請輸入一則新聞標題：", "")
threshold = st.slider("模組預測門檻", 0.1, 0.9, 0.4, step=0.05)
show_top_only = st.toggle("🔎 只顯示最高機率模組", value=False)

if user_input:
    tfidf, model, le = load_model()
    preds = predict_modules(user_input, tfidf, model, le, threshold)

    if show_top_only and preds:
        preds = [preds[0]]

    if preds:
        st.subheader("📊 模組預測圖")
        chart_df = pd.DataFrame(preds)
        chart = alt.Chart(chart_df).mark_bar(size=25).encode(
            x=alt.X('prob:Q', title='機率', scale=alt.Scale(domain=[0, 1])),
            y=alt.Y('module:N', sort='-x', title='模組'),
            color=alt.value('#1f77b4')
        ) + alt.Chart(chart_df).mark_text(
            align='left', baseline='middle', dx=3
        ).encode(
            x='prob:Q', y='module:N', text='prob:Q'
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("⚠️ 無符合門檻的模組，請調整設定或更換輸入。")

    if st.button("⚡ 執行模擬模組觸發流程"):
        df_sim = simulate_round_trace(preds)

        st.subheader("📋 模組觸發流程")
        st.dataframe(df_sim)

        st.subheader("⏰ 模組觸發時間軸")
        st.altair_chart(plot_timeline(df_sim), use_container_width=True)

        st.subheader("🔎 模擬詮釋")
        if not df_sim.empty:
            interpretation = f"""根據輸入的新聞標題，系統預測最可能起始模組為：
**{', '.join([m['module'] for m in preds])}**

透過模組間的觸發鏈模擬，共觸發 **{df_sim.shape[0]}** 個模組，歷經 **{df_sim['time'].max() + 1}** 個階段。

模擬事件摘要如下：
"""
            st.markdown(interpretation)
            for _, row in df_sim.iterrows():
                st.markdown(
                    f"- 模組 **{row['module']}** 在第 {row['time']} 階段被觸發，"
                    f"由 `{row['triggered_by']}` 引起，預估觸發機率為 `{row['probability']:.2f}`。"
                )
