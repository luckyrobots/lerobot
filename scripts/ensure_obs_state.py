import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

ROOT = Path('dataset/session_2025-10-27_10-30-43/data/chunk-000')

updated = 0
for f in sorted(ROOT.glob('file-*.parquet')):
    table = pq.read_table(f)
    cols = table.schema.names
    if 'observation.state' in cols:
        continue
    if 'action' not in cols:
        raise RuntimeError(f"Neither 'observation.state' nor 'action' in {f}")
    df = table.to_pandas()
    df['observation.state'] = df['action']
    df.to_parquet(f, index=False)
    updated += 1
    print('Updated', f)

print('Done. Files updated:', updated)
