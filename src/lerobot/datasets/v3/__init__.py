"""LeRobot dataset v3 (sharded session) support.

LeRobot v3 sessions store:
- parquet shards: data/chunk-XXX/file-YYY.parquet
- mp4 shards per video key: videos/{video_key}/chunk-XXX/file-YYY.mp4
- parquet metadata: meta/tasks.parquet, meta/episodes/**.parquet

This module provides a lightweight reader + Dataset implementation compatible with the
existing LeRobot training pipeline.
"""

from .lerobot_dataset_v3 import LeRobotDatasetMetadataV3, LeRobotDatasetV3


