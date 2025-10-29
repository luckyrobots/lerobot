import json, os
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
root = Path('dataset/session_2025-10-28_16-25-43')
info_path = root/'meta/info.json'
# compute frames and episodes from data parquet
rows = 0
unique_eps = set()
for p in sorted((root/'data').rglob('*.parquet')):
    try:
        # fast: metadata rows
        md = pq.read_metadata(p)
        rows += md.num_rows
        # get unique episodes per file cheaply
        try:
            tbl = pq.read_table(p, columns=['episode_index'])
            unique_eps.update(set(tbl['episode_index'].to_pylist()))
        except Exception:
            pass
    except Exception as e:
        print('ERR data parquet', p, e)
# count videos
video_files = list((root/'videos').rglob('*.mkv'))
# load current info
with open(info_path, 'r') as f:
    info = json.load(f)
info['total_frames'] = int(rows)
info['total_episodes'] = int(len(unique_eps)) if unique_eps else info.get('total_episodes', 0)
info['total_videos'] = int(len(video_files))
# adjust splits if test exceeds total_episodes
spl = info.get('splits', {})
train_s = spl.get('train')
if isinstance(train_s, str) and ':' in train_s:
    a,b = train_s.split(':')
    try:
        a_i,b_i = int(a), int(b)
        if b_i>info['total_episodes']:
            spl['train'] = f"{a_i}:{info['total_episodes']}"
    except Exception:
        pass
test_s = spl.get('test')
if isinstance(test_s, str) and ':' in test_s:
    a,b = test_s.split(':')
    try:
        a_i,b_i = int(a), int(b)
        if a_i>=info['total_episodes']:
            spl['test'] = f"{info['total_episodes']}:{info['total_episodes']}"
        elif b_i>info['total_episodes']:
            spl['test'] = f"{a_i}:{info['total_episodes']}"
    except Exception:
        pass
info['splits'] = spl
# write back
with open(info_path, 'w') as f:
    json.dump(info, f, indent=1)
print('Updated info.json with:', {'total_frames': info['total_frames'], 'total_episodes': info['total_episodes'], 'total_videos': info['total_videos'], 'splits': info['splits']})
