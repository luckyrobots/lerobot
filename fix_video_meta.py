import json
from pathlib import Path
import pandas as pd
import av

root = Path(r'dataset/session_2025-10-28_16-25-43')
info = json.loads((root/'meta/info.json').read_text())
fps = info['fps']
features = info['features']

# Collect video camera keys
cam_keys = [k for k,v in features.items() if v.get('dtype')=='video']

# Load episodes meta (single file after earlier fix)
meta_path = root/'meta/episodes/chunk-000/file-000.parquet'
df = pd.read_parquet(meta_path)

# Determine episode order (use dataset_from_index if present; fallback to episode_index)
order_col = 'dataset_from_index' if 'dataset_from_index' in df.columns else 'episode_index'
df = df.sort_values(order_col).reset_index(drop=True)

# For each camera, gather existing files and durations
cam_files = {}
for cam in cam_keys:
    cam_dir = root / 'videos' / cam
    files = sorted(cam_dir.rglob('file-*.mkv'))
    if not files:
        continue
    seq = []
    for p in files:
        # parse chunk and file indices from path
        try:
            chunk_idx = int(p.parent.name.split('-')[-1])  # chunk-000
        except Exception:
            chunk_idx = 0
        try:
            file_idx = int(p.name.split('-')[-1].split('.')[0])
        except Exception:
            file_idx = 0
        # duration via PyAV
        with av.open(str(p)) as c:
            v = c.streams.video[0]
            if v.duration is not None:
                dur = float(v.duration * v.time_base)
            else:
                dur = float(c.duration/av.time_base)
        seq.append({'path':p, 'chunk_index':chunk_idx, 'file_index':file_idx, 'duration':dur})
    cam_files[cam] = sorted(seq, key=lambda x: (x['chunk_index'], x['file_index']))

# Recompute mapping and timestamps
for cam in cam_keys:
    files = cam_files.get(cam, [])
    if not files:
        continue
    fptr = 0
    t_in_file = 0.0
    # prepare columns
    col_chunk = f'videos/{cam}/chunk_index'
    col_file = f'videos/{cam}/file_index'
    col_from = f'videos/{cam}/from_timestamp'
    col_to   = f'videos/{cam}/to_timestamp'
    for col in [col_chunk, col_file, col_from, col_to]:
        if col not in df.columns:
            df[col] = None
    for i,row in df.iterrows():
        ep_len = int(row['length'])
        ep_dur = ep_len / fps
        # advance to next file if current would overflow
        if t_in_file + ep_dur > files[fptr]['duration'] + 1e-6:
            fptr += 1
            t_in_file = 0.0
        if fptr >= len(files):
            raise RuntimeError(f"Ran out of video files for {cam}; need more capacity.")
        df.at[i, col_chunk] = files[fptr]['chunk_index']
        df.at[i, col_file]  = files[fptr]['file_index']
        df.at[i, col_from]  = t_in_file
        df.at[i, col_to]    = t_in_file + ep_dur
        t_in_file += ep_dur

# Save back
df.to_parquet(meta_path, index=False)
print('Updated meta with per-file timestamps for cams:', cam_keys)
