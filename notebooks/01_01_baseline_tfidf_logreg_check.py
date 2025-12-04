# baseline_tfidf_logreg_check.py
#データチェック用
#test.csvのデータをUTF-8で読み込んだ際にエラーになる行と原因を抽出
# tarinでカテゴリーがNANになって、抜け落ちているところがわかるようにする

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
OUTPUT_DIR = "outputs/proceeding"

# 元々の submission 用（今回は使っても使わなくてもOK）
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "proceeding_latin.csv")

# 追加：確認用の出力パス
UTF8_ERROR_LINES_PATH = os.path.join(OUTPUT_DIR, "test_utf8_error_lines.csv")
MISSING_CATEGORY_PATH = os.path.join(OUTPUT_DIR, "train_missing_category.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2-A. test.csv を UTF-8 で読んだときのエラー行を特定
# =========================
print("Checking UTF-8 decoding errors in test.csv ...")

utf8_error_rows = []

# バイナリで1行ずつ読み込み、UTF-8デコードを試す
with open(TEST_PATH, "rb") as f:
    for i, line in enumerate(f, start=1):
        try:
            line.decode("utf-8")
        except UnicodeDecodeError as e:
            # 問題行の一部を latin1 などで無理やり文字列にしてプレビュー
            preview = line[:200].decode("latin1", errors="replace")
            utf8_error_rows.append({
                "line_number": i,
                "error": str(e),
                "raw_preview_latin1": preview
            })

# 結果をCSVに保存
if utf8_error_rows:
    df_errors = pd.DataFrame(utf8_error_rows)
    df_errors.to_csv(UTF8_ERROR_LINES_PATH, index=False)
    print(f"  UTF-8 デコードに失敗した行が {len(utf8_error_rows)} 行あります。")
    print(f"  詳細は {UTF8_ERROR_LINES_PATH} に出力しました。")
else:
    print("  UTF-8 デコードエラーは検出されませんでした。")

# =========================
# 2-B. train.csv の読み込み & category 欠損行の保存
# =========================
print("Loading train.csv ...")
train_raw = pd.read_csv(TRAIN_PATH)

# category が NaN の行だけ抜き出し
missing_category = train_raw[train_raw["category"].isna()]

print(f"  category が NaN の行数: {len(missing_category)}")

if len(missing_category) > 0:
    missing_category.to_csv(MISSING_CATEGORY_PATH, index=False)
    print(f"  これらの行を {MISSING_CATEGORY_PATH} に保存しました。")

# ここから先は、必要なら従来どおりの前処理・学習を続けてもOK
# =========================
# 3. モデル用にデータ読み込み（従来どおり）
# =========================
print("Loading data for modeling...")

# train はさきほど読み込んだ train_raw を使ってもいいし、改めて読み直してもOK
train = train_raw.copy()

# test はこれまで通り latin1 で読んでいる（あとで UTF-8 に切り替えるかは検討）
test = pd.read_csv(TEST_PATH, encoding="latin1")

# category 欠損行を落とす
train = train.dropna(subset=["category"]).reset_index(drop=True)
print("After dropping NaN labels, train size:", len(train))

# 必要なカラムがあるか一応チェック
required_train_cols = {"cleaned_text", "category"}
required_test_cols = {"text"}

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

