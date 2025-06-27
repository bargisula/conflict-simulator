import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report

def load_and_flatten(input_csv):
    df = pd.read_csv(input_csv)
    df = df.explode("modules")
    df = df[df["modules"] != "UNK"]
    df = df.dropna(subset=["title", "modules"])
    return df[["title", "modules"]]

def train_model(df):
    X_text = df["title"]
    y = df["modules"]

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X = tfidf.fit_transform(X_text)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)

    print("📊 模型交叉驗證（5-fold）...")
    scores = cross_val_score(clf, X, y_encoded, cv=5, scoring="f1_macro")
    print(f"✅ 平均 F1 (macro): {scores.mean():.4f} ± {scores.std():.4f}")

    y_pred = cross_val_predict(clf, X, y_encoded, cv=5)
    print("\n📋 分類報告（各模組精度）:")
    print(classification_report(y_encoded, y_pred, target_names=le.classes_))

    print("\n📈 模組分佈：")
    print(df["modules"].value_counts())

    print(f"\n📦 樣本數：{len(df)}, 類別數：{len(le.classes_)}")

    # 訓練最終模型（不用拆折）
    clf.fit(X, y_encoded)
    return tfidf, clf, le

def save_models(tfidf, clf, le, output_dir="../models"):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(tfidf, os.path.join(output_dir, "tfidf.pkl"))
    joblib.dump(clf, os.path.join(output_dir, "model.pkl"))
    joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))
    print("\n✅ 模型已儲存至 models/ 資料夾")

if __name__ == "__main__":
    input_csv = "../data/newsapi_labeled.csv"
    df = load_and_flatten(input_csv)
    tfidf, clf, le = train_model(df)
    save_models(tfidf, clf, le)
