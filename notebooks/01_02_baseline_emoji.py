#UTF=8でerroes=ignoreで読み込んだ後に絵文字が問題ないかを確認したい。
import pandas as pd
import os

TEST_PATH = "data/raw/test.csv"
OUTPUT_PATH = "outputs/proceeding/test_utf8_ignore.csv"

os.makedirs("outputs/proceeding", exist_ok=True)

print("Reading test.csv with UTF-8 + errors='ignore' ...")

# open() を使って UTF-8 + ignore で読み込む
with open(TEST_PATH, "r", encoding="utf-8", errors="ignore") as f:
    test = pd.read_csv(f)

print("Loaded test.csv")

# 先頭20行表示（確認用）
print("\n--- データ先頭20行 ---")
print(test.head(20))

print(f"\nSaving UTF-8 processed CSV to: {OUTPUT_PATH}")

# UTF-8 で保存（文字化けせずそのまま確認できる）
test.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("Saved! Now you can inspect the file in a text editor or spreadsheet.")
