# baseline_tfidf_logreg.py
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# =========================
# 1. パス設定
# =========================
TRAIN_PATH = "data/raw/train.csv"
TEST_PATH = "data/raw/test.csv"
OUTPUT_DIR = "outputs/submissions"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "submission_tfidf_logreg.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2. データ読み込み
# =========================
print("Loading data...")

train_raw = pd.read_csv(
    TRAIN_PATH,
    encoding="utf-8",
    encoding_errors="ignore"
)
print("Original train size:", len(train_raw))
test = pd.read_csv(
    TEST_PATH,
    encoding="utf-8",
    encoding_errors="ignore"
)
print("Original test size:", len(test))

train = train_raw.copy()

train = train.dropna(subset=["category"]).reset_index(drop=True)
print("After dropping NaN labels, train size:", len(train))

# 必要なカラムがあるか一応チェック
required_train_cols = {"cleaned_text", "category"}
required_test_cols = { "text"}

if not required_train_cols.issubset(train.columns):
    raise ValueError(f"train.csv に必要なカラムが足りません: {required_train_cols - set(train.columns)}")

if not required_test_cols.issubset(test.columns):
    raise ValueError(f"test.csv に必要なカラムが足りません: {required_test_cols - set(test.columns)}")

# テキストとターゲット
X = train["cleaned_text"].fillna("")
y = train["category"]

test["cleaned_text"] = test["text"].fillna("")
X_test_text = test["cleaned_text"]
test_ids = test["ID"]

print(f"Train size: {len(train)}, Test size: {len(test)}")
print("Sample categories:", y.unique())

# =========================
# 3. 学習用の簡易バリデーション
#    train を train/valid に分割して F1 を見てみる
# =========================
X_train_text, X_valid_text, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 4. TF-IDF ベクトル化
# =========================
print("Vectorizing text with TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=30000,   # 語彙数の上限（まずは3万くらいでOK）
    ngram_range=(1, 2),   # uni-gram + bi-gram
    min_df=2,             # 2文書未満しか出ない単語は無視
)

X_train_vec = vectorizer.fit_transform(X_train_text)
X_valid_vec = vectorizer.transform(X_valid_text)
X_test_vec = vectorizer.transform(X_test_text)

print("TF-IDF shapes:")
print("  Train:", X_train_vec.shape)
print("  Valid:", X_valid_vec.shape)
print("  Test :", X_test_vec.shape)

# =========================
# 5. ロジスティック回帰で学習
# =========================
print("Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=2000,
    n_jobs=-1,
    C=1.0,
    verbose=0,
)

model.fit(X_train_vec, y_train)

# =========================
# 6. バリデーション評価（F1 Micro）
# =========================
print("Evaluating on validation set...")
valid_pred = model.predict(X_valid_vec)
f1_micro = f1_score(y_valid, valid_pred, average="micro")
print(f"Validation F1 (micro): {f1_micro:.4f}")

# =========================
# 7. 全データで学習し直して test を予測（任意）
#    ※ バリデーションでの様子を見たあと、
#       本番用に train 全体で学習し直す。
# =========================
print("Training final model on full training data...")

X_full_vec = vectorizer.fit_transform(X)
X_test_vec_full = vectorizer.transform(X_test_text)

final_model = LogisticRegression(
    max_iter=2000,
    n_jobs=-1,
    C=1.0,
    verbose=0,
)

final_model.fit(X_full_vec, y)

test_pred = final_model.predict(X_test_vec_full)

# =========================
# 8. 提出ファイル作成
# =========================
print(f"Saving submission to {OUTPUT_PATH} ...")

submission = pd.DataFrame({
    "id": test["ID"],
    "category": test_pred
})

submission.to_csv(OUTPUT_PATH, index=False)

print("Done!")
