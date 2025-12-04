import pandas as pd
import os

TRAIN_PATH = "data/raw/train.csv"
OUTPUT_DIR = "outputs/proceeding"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "text_diff.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# train 読み込み
train = pd.read_csv(TRAIN_PATH)

# NaN 対策
orig = train["original_text"].fillna("")
clean = train["cleaned_text"].fillna("")

# original_text と cleaned_text が違う行を抽出
diff_mask = orig != clean
diff_rows = train[diff_mask][["original_text", "cleaned_text"]]

print("original_text と cleaned_text が違う行数:", len(diff_rows))
print(diff_rows.head(20))

# 差分を CSV に保存
diff_rows.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"\n差分データを {OUTPUT_PATH} に保存しました。")
