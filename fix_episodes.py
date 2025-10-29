import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from collections import Counter
root = Path('dataset/session_2025-10-28_16-25-43')
# load existing episodes meta
meta_dir = root/'meta/episodes/chunk-000'
parts = sorted(meta_dir.glob('file-*.parquet'))
dfs = []
for p in parts:
    try:
        df = pd.read_parquet(p)
        dfs.append(df)
    except Exception as e:
        print('ERR reading', p, e)
if not dfs:
    print('No episodes parquet found')
    raise SystemExit(0)
meta = pd.concat(dfs, ignore_index=True)
# compute actual per-episode lengths from data parquet
counts = Counter()
for p in sorted((root/'data').rglob('*.parquet')):
    try:
        tbl = pq.read_table(p, columns=['episode_index'])
        for v in tbl['episode_index'].to_pylist():
            counts[int(v)] += 1
    except Exception as e:
        print('ERR data parquet', p, e)
if not counts:
    print('No counts from data; abort')
    raise SystemExit(1)
# collapse duplicates keeping first occurrence for each episode_index to preserve video/file idx columns
meta = meta.sort_values(['episode_index']).drop_duplicates(subset=['episode_index'], keep='first').reset_index(drop=True)
# replace length with real counts, dropping episodes not present in data
rows = []
episode_order = sorted(counts.keys())
cum = 0
for ep in episode_order:
    row = meta.loc[meta['episode_index']==ep]
    if row.empty:
        # create a minimal row if not present in meta
        row = pd.DataFrame({'episode_index':[ep]})
    row = row.iloc[0].to_dict()
    length = counts[ep]
    row['length'] = length
    row['dataset_from_index'] = cum
    row['dataset_to_index'] = cum + length
    cum += length
    rows.append(row)
fixed = pd.DataFrame(rows)
# ensure stable column ordering: keep original columns first
cols = list(meta.columns)
for c in fixed.columns:
    if c not in cols:
        cols.append(c)
fixed = fixed.reindex(columns=cols)
# write back into a single parquet file (overwrite file-000.parquet)
out = meta_dir/'file-000.parquet'
fixed.to_parquet(out, index=False)
# remove any extra episode parquet files to avoid stale duplicates
for p in parts:
    if p != out:
        try:
            p.unlink()
        except Exception:
            pass
print('Wrote', out, 'rows:', len(fixed), 'last_to:', int(fixed['dataset_to_index'].iloc[-1]))
