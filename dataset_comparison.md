# LeRobot Dataset Format 3.0 — Graphical Reference

ASCII Directory Layout (3.0) with 10 episodes per file

```
dataset/
├── data/
│   └── chunk-000/
│       └── file-000.parquet  # contains 10 episodes
├── videos/
│   └── {camera_key}/chunk-000/file-000.mp4  # contains 10 episodes per file
├── meta/
│   ├── episodes/
│   │   └── chunk-000/
│   │       └── file-000.parquet  # 10 episodes per file
│   ├── info.json
│   ├── stats.json
│   └── tasks/
│       └── chunk-000/
│           └── file_000.parquet  # per-chunk tasks for 10 episodes
```

Directory Layout (detailed, 3.0)

- `data/`: chunked Parquet shards, e.g. `data/chunk-000/file-000.parquet` (10 episodes per file)
- `videos/`: per-camera MP4 shards, e.g. `videos/{camera_key}/chunk-000/file_000.mp4` (10-episode groups)
- `meta/`:
  - `episodes/chunk-000/file-000.parquet` (per-chunk episode metadata; 10 episodes per file)
  - `info.json` (canonical schema, FPS, path templates)
  - `stats.json` (global feature statistics)
  - `tasks/chunk-000/file_000.parquet` (per-chunk task mappings for 10 episodes)

File Content Examples (3.0)

meta/info.json
```json
{
  "codebase_version": "v3.0",
  "robot_type": "so100_dualcam_pick_place",
  "fps": 30,
  "splits": {"train": "0:1"},
  "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
  "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
  "features": {
    "timestamp": {"dtype": "float64"},
    "action": {"dtype": "float32", "shape": [ /* variable by task */ ]},
    "observation": {
      "state": {"dtype": "float32"},
      "images": {"dtype": "bytes"}
    }
  }
}
```

meta/episodes/chunk-000/file-000.parquet (textual schema)
```
episode_index | video_chunk_index | video_file_index | data_chunk_index | data_file_index | tasks | length
0             | 0               | 0              | 0              | 0             | [0]   | 256
1             | 0               | 0              | 0              | 0             | [0]   | 312
```

meta/tasks/chunk-000/file_000.parquet (textual schema)
```
task_index | task
---------- | --------------------------------
0          | "Pick and place blue cube into target zone"
```

data/chunk-000/file-000.parquet (textual schema)
```
timestamp | observation.state | observation.images.front | action
0.000     | [0.0, 0.0, 0.0]  | <image bytes>           | [0.0, 0.0, 0.0]
```

videos/chunk-000/file_000.mp4 (descriptor)
```
MP4 container: frames for each camera key, per-chunk shard
- camera: observation.images.front
- camera: observation.images.side
```

Loader expectations (3.0)
- Read per-chunk episode metadata from `meta/episodes/chunk-000/file-000.parquet` to enumerate episodes
- Read per-chunk tasks from `meta/tasks/chunk-000/file_000.parquet` for task conditioning
- Load data shards from `data/chunk-000/file-000.parquet` and decode with the corresponding video shards from `videos/{camera_key}/chunk-000/file_000.mp4`
- Use templates in `meta/info.json` to map episodes to their data/video shards

Notes
- This 3.0 layout is designed to scale by grouping episodes into large, shared Parquet and MP4 files and resolving episode boundaries via metadata.
- The per-chunk tasks Parquet is the canonical layout (e.g., `meta/tasks/chunk-000/file_000.parquet`).

Info.json, Stats.json and Tasks deeper dive

info.json details
```json
{
  "codebase_version": "v3.0",
  "robot_type": "so100_dualcam_pick_place",
  "fps": 30,
  "splits": {
    "train": "0:1",
    "val": "1:1"
  },
  "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
  "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
  "features": {
    "timestamp": {"dtype": "float64"},
    "action": {"dtype": "float32", "shape": [3]},
    "observation": {
      "state": {"dtype": "float32", "shape": [6]},
      "images": {"dtype": "bytes"}
    }
  },
  "version": 1
}
```

stats.json example
```json
{
  "timestamp": {"mean": 0.012, "std": 0.003, "min": -0.02, "max": 0.05},
  "observation.state": {
    "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "std":  [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    "min": [-1, -1, -1, -1, -1, -1],
    "max": [ 1,  1,  1,  1,  1,  1]
  },
  "action": {
    "mean": [0.0, 0.0, 0.0],
    "std": [0.5, 0.5, 0.5],
    "min": [-1, -1, -1],
    "max": [ 1,  1,  1]
  }
}
```

meta/tasks/chunk-000/file_000.parquet (textual schema)
```text
task_index | task
---------- | --------------------------------
0          | "Pick and place blue cube into target zone"
```

## Robot type mapping

In info.json, `robot_type` must reflect the actual robot used to collect the dataset. For So100 datasets, use `so100_*` slugs such as `so100_dualcam_pick_place` or `so100_shaver_insert`. For datasets from other robots, use the corresponding robot slug used during recording.

## Actuator/Action dimension variability

- The number of action dimensions (and corresponding actuator signals) depends on the robot type used to collect the dataset.
- Check `meta/info.json` for the `robot_type` and the `features.action` shape to determine the exact dimensionality.
- For So100 datasets (e.g., `so100_dualcam_pick_place`, `so100_shaver_insert`), action shapes are defined under `features.action` (dtype and shape) in `info.json`.
- If `robot_type` differs, refer to the corresponding section in `features` for that robot to determine the action dimensionality.
