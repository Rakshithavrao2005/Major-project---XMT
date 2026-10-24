import os, json, pandas as pd, pyarrow as pa
root = r'C:\Users\Varshini\.gemini\antigravity\scratch\XMT-Sepsis-System\data\processed\hourly_cache\_staging_tmp'
files = {
    'train': {
        'labels': 'labels_train.parquet',
        'vitals': 'vitals_train.parquet',
        'labs': 'labs_train.parquet',
    },
    'val': {
        'labels': 'labels_val.parquet',
        'vitals': 'vitals_val.parquet',
        'labs': 'labs_val.parquet',
    },
    'test': {
        'labels': 'labels_test.parquet',
        'vitals': 'vitals_test.parquet',
        'labs': 'labs_test.parquet',
    },
}
result = {}
for split, parts in files.items():
    result[split] = {}
    for typ, fname in parts.items():
        path = os.path.join(root, fname)
        df = pd.read_parquet(path)
        # schema via pyarrow
        schema = pa.parquet.read_schema(path)
        stay_ids = df['stay_id'].unique()
        dup_keys = df.duplicated(subset=['stay_id']).sum()
        missing_stay = df['stay_id'].isna().sum()
        # For labels, distribution of label_status
        dist = None
        if typ == 'labels':
            dist = df['label_status'].value_counts(dropna=False).to_dict()
            # hourly coverage per stay: count rows per stay_id
            coverage = df.groupby('stay_id').size().describe().to_dict()
        else:
            coverage = None
        result[split][typ] = {
            'rows': len(df),
            'schema': [f"{field.name}: {field.type}" for field in schema],
            'unique_stay_ids': int(len(stay_ids)),
            'duplicate_rows': int(dup_keys),
            'missing_stay_id': int(missing_stay),
            'label_status_distribution': dist,
            'stay_hourly_coverage_stats': coverage,
        }
print(json.dumps(result, indent=2))
