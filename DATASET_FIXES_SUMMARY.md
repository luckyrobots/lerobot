# Dataset Fixes Summary: session_2025-10-27_10-30-43

## Overview
Fixed dataset schema mismatches, missing metadata columns, and feature dimension inconsistencies to enable training.

---

## 1. meta/info.json - Feature Configuration

### Action Feature
| Aspect | Current | Expected |
|--------|---------|----------|
| shape | [6] | [8] |
| names (count) | 6 names | 8 names |
| names (values) | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper | + aux_0, aux_1 |

### Observation State Feature
| Aspect | Current | Expected |
|--------|---------|----------|
| shape | [6] | [8] |
| names (count) | 6 names | 8 names |
| names (values) | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper | + aux_0, aux_1 |

### Video Camera Keys
| Feature Key | Current | Expected | Reason |
|------------|---------|----------|--------|
| observation.images.camera | ✓ exists | ✗ remove | Doesn't match video folder structure |
| observation.images.camera_2 | ✓ exists | ✗ remove | Doesn't match video folder structure |
| observation.images.camera_0 | ✗ missing | ✓ add | Matches videos/observation.images.camera_0/ |
| observation.images.camera_1 | ✗ missing | ✓ add | Matches videos/observation.images.camera_1/ |

---

## 2. meta/episodes/chunk-000/*.parquet - Episode Metadata (3 files)

### Data Path Columns
| Column | Current | Expected | Computation |
|--------|---------|----------|-------------|
| data/chunk_index | ✗ missing | ✓ int64 | Copied from data_chunk_index |
| data/file_index | ✗ missing | ✓ int64 | Copied from data_file_index |

### Episode Frame Range Columns
| Column | Current | Expected | Computation |
|--------|---------|----------|-------------|
| dataset_from_index | ✗ missing | ✓ int64 | min(index) per episode in data files |
| dataset_to_index | ✗ missing | ✓ int64 | max(index) + 1 per episode in data files |

### Camera 0 Video Path Columns
| Column | Current | Expected | Source |
|--------|---------|----------|--------|
| videos/observation.images.camera_0/chunk_index | ✗ missing | ✓ int64 | Copied from video_chunk_index |
| videos/observation.images.camera_0/file_index | ✗ missing | ✓ int64 | Copied from video_file_index |
| videos/observation.images.camera_0/from_timestamp | ✗ missing | ✓ float64 | Cumulative duration per (chunk, file) group |
| videos/observation.images.camera_0/to_timestamp | ✗ missing | ✓ float64 | from_timestamp + (frame_count / fps=30) |

### Camera 1 Video Path Columns
| Column | Current | Expected | Source |
|--------|---------|----------|--------|
| videos/observation.images.camera_1/chunk_index | ✗ missing | ✓ int64 | Copied from video_chunk_index |
| videos/observation.images.camera_1/file_index | ✗ missing | ✓ int64 | Copied from video_file_index |
| videos/observation.images.camera_1/from_timestamp | ✗ missing | ✓ float64 | Cumulative duration per (chunk, file) group |
| videos/observation.images.camera_1/to_timestamp | ✗ missing | ✓ float64 | from_timestamp + (frame_count / fps=30) |

---

## 3. data/chunk-000/*.parquet - Data Records (3 files)

### Missing Observation Column
| Column | Current | Expected | Type | Source |
|--------|---------|----------|------|--------|
| observation.state | ✗ missing | ✓ required | list<float> (8 elements) | Copied from action |

---

## Remaining Issue: Video Backend

| Aspect | Current | Expected | Solution |
|--------|---------|----------|----------|
| Codec Support | torchcodec fails on MKV | Use alternative backend | `--dataset.video_backend=av` |
| Video Format (alt) | MKV files | Remux to MP4 (no re-encode) | `ffmpeg -i video.mkv -c copy video.mp4` |

**Root Cause:** torchcodec cannot read max PTS metadata from MKV stream headers.


