import pandas as pd, os, json, sys
root = r'C:\Users\Varshini\.gemini\antigravity\scratch\XMT-Sepsis-System\data\processed\hourly_cache\_staging_tmp'
files = ['labels_train.parquet','labels_val.parquet','labels_test.parquet','vitals_train.parquet','vitals_val.parquet','vitals_test.parquet','labs_train.parquet','labs_val.parquet','labs_test.parquet']
counts = {}
for f in files:
    path = os.path.join(root, f)
    if not os.path.exists(path):
        counts[f] = None
        continue
    df = pd.read_parquet(path)
    counts[f] = df.shape[0]
print(json.dumps(counts, indent=2))
