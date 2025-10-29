import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import av

root = Path(r'dataset/session_2025-10-28_16-25-43')
fps = json.loads((root/'meta/info.json').read_text())['fps']
cam = 'observation.images.camera_0'
vid = root/f'videos/{cam}/chunk-000/file-000.mkv'
meta_ep_path = root/'meta/episodes/chunk-000/file-000.parquet'
data_path = root/'data/chunk-000/file-000.parquet'

print('FPS:', fps)
print('Video:', vid)

# Video info via PyAV
try:
    with av.open(str(vid)) as c:
        v = c.streams.video[0]
        print('PyAV base_rate:', float(v.base_rate))
        print('PyAV r_frame_rate:', v.rate if hasattr(v,'rate') else 'NA')
        print('PyAV time_base:', v.time_base)
        print('PyAV frames (may be 0 if unknown):', getattr(v, 'frames', None))
        dur = (float(v.duration * v.time_base) if v.duration is not None else float(c.duration/av.time_base))
        print('PyAV duration_s:', dur)
except Exception as e:
    print('PyAV open error:', e)

# Try torchcodec metadata (best-effort)
try:
    import importlib
    if importlib.util.find_spec('torchcodec'):
        from torchcodec.decoders import VideoDecoder
        dec = VideoDecoder(str(vid), seek_mode='exact')
        md = dec.metadata
        attrs = {k:getattr(md,k) for k in dir(md) if not k.startswith('_')}
        print('torchcodec.metadata keys:', sorted(attrs.keys()))
        for key in ['average_fps','duration_seconds','num_frames','frames','frame_count']:
            print('torchcodec', key, ':', getattr(md, key, getattr(dec, key, None)))
    else:
        print('torchcodec not available')
except Exception as e:
    print('torchcodec init/metadata error:', e)

# Parquet frame timestamps
try:
    tbl = pq.read_table(data_path, columns=['timestamp','episode_index'])
    ts = tbl['timestamp'].to_pylist()
    eps = tbl['episode_index'].to_pylist()
    print('Parquet rows:', len(ts))
    if ts:
        print('Parquet ts[0:3]:', ts[:3])
        print('Parquet ts[-3:]:', ts[-3:])
        # deltas sample
        import numpy as np
        arr = np.array(ts, dtype=float)
        d = np.diff(arr)
        if d.size:
            print('Parquet dt min/max/mean:', float(d.min()), float(d.max()), float(d.mean()))
except Exception as e:
    print('Parquet read error:', e)

# Episodes meta for this camera/file
try:
    ep = pd.read_parquet(meta_ep_path)
    mask = (ep[f'videos/{cam}/chunk_index']==0) & (ep[f'videos/{cam}/file_index']==0)
    sel = ep.loc[mask, ['episode_index','length', f'videos/{cam}/from_timestamp', f'videos/{cam}/to_timestamp']]
    sel = sel.sort_values('episode_index')
    print('Episodes mapped to this mkv (count):', len(sel))
    if len(sel):
        print(sel.head(min(5,len(sel))))
        total_len = int(sel['length'].sum())
        total_dur = float(total_len / fps)
        last_to = float(sel[f'videos/{cam}/to_timestamp'].max())
        print('Sum length:', total_len, 'frames ->', total_dur, 's; max to_timestamp:', last_to)
        # check end-of-last-episode timestamp
        last = sel.iloc[-1]
        end_ts = float(last[f'videos/{cam}/from_timestamp'] + (last['length']-1)/fps)
        print('Computed last frame ts:', end_ts)
except Exception as e:
    print('Meta episodes read error:', e)
