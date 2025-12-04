## 2025-12-02 (Baseline)

初回ベースライン
01_baseline_tfidf_logreg.py
Model: TF-IDF + LR
Validation F1: 0.9283
LB Score: 0.74266
Notes: Cleaned text only. No tuning.

調整メモ：
01_01_baseline_tfidf_logreg_check.py
01_02_baseline_emoji.py
・train.csv は 152499 個。test.csv は 1500 個。
・test.csv で UTF-8 で読み込めないのは 12 個程度で、重要な記号はない。
・NAN やエラーになっているデータは 6435 個。ほとんどがデータなし。80 個が original のみ。
・original と cleaned に違いがあるのが 3025 個。(小文字 → 大文字、can't→cannot)
=== is_modified の割合 (%) ===
is_modified
True 59.75
False 36.03
NaN 4.22
まとめ：懸念点は AI っぽい綺麗な文で主に学習 → 本番は X の生テキストだが、
　　　　モデルである程度解決できそう。また、それ以外は影響度が小さいので、今回はむし。

2 回目データ処理の変更
02_baseline_dataprocessing.py
Model: TF-IDF + LR
Validation F1: 0.9283
LB Score: 0.74266
Notes:utf-8 に変更の身。

3 回目
