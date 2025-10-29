import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

root = Path('dataset/session_2025-10-27_10-30-43/data/chunk-000')
cols = ['action','observation.state','index','episode_index']

for f in sorted(root.glob('file-*.parquet')):
    table = pq.read_table(f)
    names = table.schema.names
    use_cols = [c for c in cols if c in names]
    df = table.select(use_cols).to_pandas()

    def has_none(series):
        return series.isna().any() or series.apply(lambda x: x is None).any()

    issues = {c: has_none(df[c]) for c in use_cols if c in df.columns}
    if any(issues.values()):
        print('File with nulls:', f)
        for k, v in issues.items():
            if v:
                print('  -', k, 'has nulls')
        # Show example rows
        bad = df[df.apply(lambda r: any([(r[c] is None) or pd.isna(r[c]) for c in use_cols if c in r]), axis=1)]
        print(bad.head(3))
