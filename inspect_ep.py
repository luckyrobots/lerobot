import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
p0 = Path('dataset/session_2025-10-28_16-25-43/meta/episodes/chunk-000/file-000.parquet')
p1 = Path('dataset/session_2025-10-28_16-25-43/meta/episodes/chunk-000/file-001.parquet')

dfs = []
for p in [p0,p1]:
    if p.exists():
        try:
            df = pd.read_parquet(p)
            df['__src']=p.name
            dfs.append(df)
        except Exception as e:
            print('ERR reading', p, e)

if not dfs:
    print('No episode parquet read')
    raise SystemExit(0)

df = pd.concat(dfs, ignore_index=True)
cols = ['episode_index','length','dataset_from_index','dataset_to_index']
print('Columns:', df.columns.tolist())
print(df[cols].head(5))
print(df[cols].tail(5))
print('n_episodes:', len(df))
print('min_from:', int(df['dataset_from_index'].min()), 'max_to:', int(df['dataset_to_index'].max()))
print('is_monotonic_from:', df['dataset_from_index'].is_monotonic_increasing)
print('is_monotonic_to:', df['dataset_to_index'].is_monotonic_increasing)
print('total_frames_sum_len:', int(df['length'].sum()))
print('last_train_to:', int(df.loc[df['episode_index']==11,'dataset_to_index'].iloc[0]) if (df['episode_index']==11).any() else 'NA')
print('last_to:', int(df['dataset_to_index'].iloc[-1]))
