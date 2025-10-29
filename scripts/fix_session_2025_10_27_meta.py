import glob
import pandas as pd
from pathlib import Path

ROOT = Path('dataset/session_2025-10-27_10-30-43/meta/episodes/chunk-000')

paths = sorted(glob.glob(str(ROOT / '*.parquet')))
print(f'Found {len(paths)} episode parquet files')

for p in paths:
    print(f'Processing {p}')
    df = pd.read_parquet(p)

    # Add data path columns expected by code
    if 'data_chunk_index' in df.columns and 'data_file_index' in df.columns:
        df['data/chunk_index'] = df['data_chunk_index']
        df['data/file_index'] = df['data_file_index']

    # Add per-video path columns for both cameras (assuming same indexing)
    if 'video_chunk_index' in df.columns and 'video_file_index' in df.columns:
        df['videos/observation.images.camera_0/chunk_index'] = df['video_chunk_index']
        df['videos/observation.images.camera_0/file_index'] = df['video_file_index']
        df['videos/observation.images.camera_1/chunk_index'] = df['video_chunk_index']
        df['videos/observation.images.camera_1/file_index'] = df['video_file_index']

    df.to_parquet(p, index=False)

print('Done.')
