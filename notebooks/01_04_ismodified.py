import pandas as pd

TRAIN_PATH = "data/raw/train.csv"

# train 読み込み
train = pd.read_csv(TRAIN_PATH)

# is_modified の件数
count = train["is_modified"].value_counts(dropna=False)

# 割合（％）
ratio = train["is_modified"].value_counts(normalize=True, dropna=False) * 100

print("=== is_modified の件数 ===")
print(count)

print("\n=== is_modified の割合 (%) ===")
print(ratio.round(2))
