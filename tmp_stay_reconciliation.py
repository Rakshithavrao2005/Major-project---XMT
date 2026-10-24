import os, sys, json, pandas as pd

# Project root
project_root = r'C:\Users\Varshini\.gemini\antigravity\scratch\XMT-Sepsis-System'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.hourly.dataset import patient_level_split

cohort_csv = r'C:\Users\Varshini\OneDrive\Downloads\cohort_icustays.csv'
cohort_df = pd.read_csv(cohort_csv)
train_stays, val_stays, test_stays = patient_level_split(cohort_df, test_ratio=0.20, val_ratio=0.15, random_state=42)

root = r'C:\Users\Varshini\.gemini\antigravity\scratch\XMT-Sepsis-System\data\processed\hourly_cache\_staging_tmp'
files = {
    'train': {'labels':'labels_train.parquet','vitals':'vitals_train.parquet','labs':'labs_train.parquet'},
    'val':   {'labels':'labels_val.parquet','vitals':'vitals_val.parquet','labs':'labs_val.parquet'},
    'test':  {'labels':'labels_test.parquet','vitals':'vitals_test.parquet','labs':'labs_test.parquet'},
}

actual_ids = {}
for split, parts in files.items():
    actual_ids[split] = {}
    for typ, fname in parts.items():
        path = os.path.join(root, fname)
        df = pd.read_parquet(path, columns=['stay_id'])
        ids = set(df['stay_id'].unique())
        actual_ids[split][typ] = ids

out = {}
for split in ['train','val','test']:
    expected = set(train_stays) if split=='train' else set(val_stays) if split=='val' else set(test_stays)
    out[split] = {}
    for typ in ['labels','vitals','labs']:
        actual = actual_ids[split][typ]
        missing = expected - actual
        extra   = actual - expected
        out[split][typ] = {
            'missing_count': len(missing),
            'extra_count': len(extra),
            'missing_example': list(missing)[:5],
            'extra_example': list(extra)[:5]
        }
print(json.dumps(out, indent=2))
