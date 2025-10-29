import pyarrow.parquet as pq
import glob

paths = sorted(glob.glob('dataset/session_2025-10-27_10-30-43/meta/episodes/chunk-000/*.parquet'))
for p in paths:
    print('FILE:', p)
    t = pq.read_table(p)
    print(t.schema)
    break
