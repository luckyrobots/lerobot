import json
from pathlib import Path
import pandas as pd

SESSION_ROOT = Path('dataset/session_2025-10-27_10-30-43')
EP_DIR = SESSION_ROOT / 'meta' / 'episodes' / 'chunk-000'
INFO_PATH = SESSION_ROOT / 'meta' / 'info.json'

with open(INFO_PATH, 'r') as f:
    info = json.load(f)

fps = info.get('fps', 30)

# Detect video keys from info.json
video_keys = [k for k, v in info['features'].items() if k.startswith('observation.images.') and v['dtype'] == 'video']
if not video_keys:
    raise SystemExit('No video keys found in info.json')

print('Video keys:', video_keys)

paths = sorted(EP_DIR.glob('file-*.parquet'))
print('Episode parquet files:', len(paths))

for p in paths:
    print('Processing', p)
    df = pd.read_parquet(p)

    required_cols = ['episode_index', 'dataset_from_index', 'dataset_to_index']
    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(f"Missing required column '{col}' in {p}")

    for vk in video_keys:
        chunk_col = f'videos/{vk}/chunk_index'
        file_col = f'videos/{vk}/file_index'
        from_col = f'videos/{vk}/from_timestamp'
        to_col = f'videos/{vk}/to_timestamp'

        if chunk_col not in df.columns or file_col not in df.columns:
            raise RuntimeError(f"Missing '{chunk_col}' or '{file_col}' in {p}")

        # Initialize columns if absent
        if from_col not in df.columns:
            df[from_col] = 0.0
        if to_col not in df.columns:
            df[to_col] = 0.0

        # For each (chunk,file) group, compute cumulative durations ordered by dataset_from_index
        grp = df.groupby([chunk_col, file_col], sort=False)
        for (chunk_idx, file_idx), g in grp:
            # Sort by dataset_from_index to respect video concatenation order
            g_sorted = g.sort_values('dataset_from_index')

            cum = 0.0
            for idx, row in g_sorted.iterrows():
                length_frames = int(row['dataset_to_index']) - int(row['dataset_from_index'])
                duration = length_frames / fps
                df.at[idx, from_col] = cum
                df.at[idx, to_col] = cum + duration
                cum += duration

    df.to_parquet(p, index=False)

print('Done.')
