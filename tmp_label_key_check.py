import pandas as pd, os, json
root = r'C:\Users\Varshini\.gemini\antigravity\scratch\XMT-Sepsis-System\data\processed\hourly_cache\_staging_tmp'
files = {
    'train': 'labels_train.parquet',
    'val':   'labels_val.parquet',
    'test':  'labels_test.parquet',
}
out = {}
for split, fname in files.items():
    path = os.path.join(root, fname)
    df = pd.read_parquet(path, columns=['stay_id','hour_index'])
    dup = df.duplicated(subset=['stay_id','hour_index']).sum()
    # continuity per stay
    continuity_issues = []
    for sid, grp in df.groupby('stay_id'):
        hrs = grp['hour_index'].sort_values().tolist()
        # check if sequence starts at 1 and increments by 1 without gaps up to max
        if hrs[0] != 1 or any(b!=a+1 for a,b in zip(hrs,hrs[1:])):
            continuity_issues.append(sid)
    out[split] = {
        'duplicate_key_count': int(dup),
        'stay_count_with_continuity_issues': len(continuity_issues),
        'example_issues': continuity_issues[:5],
    }
print(json.dumps(out, indent=2))
