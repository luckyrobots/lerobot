from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class LeRobotV3VideoSpan:
    video_key: str
    chunk_index: int
    file_index: int
    from_timestamp_s: float
    to_timestamp_s: float


@dataclass(frozen=True)
class LeRobotV3EpisodeSpan:
    """Episode metadata for mapping parquet rows to sharded video files in v3 sessions."""

    episode_index: int
    data_chunk_index: int
    data_file_index: int
    dataset_from_index: int
    dataset_to_index: int
    length_frames: int
    videos: Dict[str, LeRobotV3VideoSpan]


@dataclass(frozen=True)
class LeRobotV3SessionInfo:
    codebase_version: str
    fps: int
    data_path_fmt: str
    video_path_fmt: str
    features: dict


class LeRobotV3Session:
    """Reader for LeRobotDataset v3.0 'file-xxx' sharded sessions."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.info = self._load_info()
        self._episodes: Optional[List[LeRobotV3EpisodeSpan]] = None
        self._episode_by_index: Optional[Dict[int, LeRobotV3EpisodeSpan]] = None
        self._task_by_index: Optional[Dict[int, str]] = None

    def _load_info(self) -> LeRobotV3SessionInfo:
        info_path = self.root / "meta" / "info.json"
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        return LeRobotV3SessionInfo(
            codebase_version=str(info.get("codebase_version", "")),
            fps=int(info["fps"]),
            data_path_fmt=str(info["data_path"]),
            video_path_fmt=str(info["video_path"]),
            features=dict(info["features"]),
        )

    @property
    def fps(self) -> int:
        return self.info.fps

    def data_parquet_path(self, *, chunk_index: int, file_index: int) -> Path:
        rel = self.info.data_path_fmt.format(chunk_index=chunk_index, file_index=file_index)
        return self.root / rel

    def video_mp4_path(self, *, video_key: str, chunk_index: int, file_index: int) -> Path:
        rel = self.info.video_path_fmt.format(video_key=video_key, chunk_index=chunk_index, file_index=file_index)
        return self.root / rel

    def load_tasks(self) -> Dict[int, str]:
        if self._task_by_index is not None:
            return self._task_by_index
        try:
            import pyarrow.parquet as pq
        except Exception as e:  # pragma: no cover
            raise ImportError("pyarrow is required to read LeRobot v3 tasks.parquet") from e

        tasks_path = self.root / "meta" / "tasks.parquet"
        table = pq.read_table(tasks_path)
        # Expected columns: task_index, task_description
        task_index = table["task_index"].to_numpy()
        task_desc = table["task_description"].to_pylist()
        mapping: Dict[int, str] = {int(i): str(d) for i, d in zip(task_index, task_desc, strict=False)}
        self._task_by_index = mapping
        return mapping

    def load_episodes(self, *, include_video_keys: Optional[Iterable[str]] = None) -> List[LeRobotV3EpisodeSpan]:
        """Load episode spans from meta/episodes parquet.

        Args:
            include_video_keys: If provided, only parse spans for these video keys.
        """
        if self._episodes is not None:
            return self._episodes
        try:
            import pyarrow.parquet as pq
        except Exception as e:  # pragma: no cover
            raise ImportError("pyarrow is required to read LeRobot v3 meta/episodes parquet") from e

        episodes_root = self.root / "meta" / "episodes"
        parquet_files = sorted(episodes_root.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No episodes parquet found under {episodes_root}")

        allowed = set(include_video_keys) if include_video_keys is not None else None
        spans: List[LeRobotV3EpisodeSpan] = []

        for fpath in parquet_files:
            table = pq.read_table(fpath)
            cols = set(table.column_names)
            required = [
                "episode_index",
                "data_chunk_index",
                "data_file_index",
                "dataset_from_index",
                "dataset_to_index",
                "length",
            ]
            missing = [c for c in required if c not in cols]
            if missing:
                raise ValueError(f"Episodes parquet {fpath} missing columns: {missing}")

            # Discover video keys from column names like:
            # videos/{video_key}/chunk_index, file_index, from_timestamp, to_timestamp
            video_keys: set[str] = set()
            for name in cols:
                if not name.startswith("videos/") or not name.endswith("/chunk_index"):
                    continue
                video_key = name[len("videos/") : -len("/chunk_index")]
                if allowed is None or video_key in allowed:
                    video_keys.add(video_key)

            # Read required scalars
            ep_idx = table["episode_index"].to_numpy()
            data_chunk = table["data_chunk_index"].to_numpy()
            data_file = table["data_file_index"].to_numpy()
            ds_from = table["dataset_from_index"].to_numpy()
            ds_to = table["dataset_to_index"].to_numpy()
            length = table["length"].to_numpy()

            # Read video spans per video key
            videos_cols: dict[str, dict[str, object]] = {}
            for vk in sorted(video_keys):
                base = f"videos/{vk}"
                needed = [f"{base}/chunk_index", f"{base}/file_index", f"{base}/from_timestamp", f"{base}/to_timestamp"]
                if not all(c in cols for c in needed):
                    # Skip partial/incomplete video keys.
                    continue
                videos_cols[vk] = {
                    "chunk": table[f"{base}/chunk_index"].to_numpy(),
                    "file": table[f"{base}/file_index"].to_numpy(),
                    "from": table[f"{base}/from_timestamp"].to_numpy(),
                    "to": table[f"{base}/to_timestamp"].to_numpy(),
                }

            for i in range(len(ep_idx)):
                videos: Dict[str, LeRobotV3VideoSpan] = {}
                for vk, arrs in videos_cols.items():
                    videos[vk] = LeRobotV3VideoSpan(
                        video_key=vk,
                        chunk_index=int(arrs["chunk"][i]),
                        file_index=int(arrs["file"][i]),
                        from_timestamp_s=float(arrs["from"][i]),
                        to_timestamp_s=float(arrs["to"][i]),
                    )
                spans.append(
                    LeRobotV3EpisodeSpan(
                        episode_index=int(ep_idx[i]),
                        data_chunk_index=int(data_chunk[i]),
                        data_file_index=int(data_file[i]),
                        dataset_from_index=int(ds_from[i]),
                        dataset_to_index=int(ds_to[i]),
                        length_frames=int(length[i]),
                        videos=videos,
                    )
                )

        spans.sort(key=lambda s: s.episode_index)
        self._episodes = spans
        self._episode_by_index = {s.episode_index: s for s in spans}
        return spans

    def episode_span(self, episode_index: int) -> LeRobotV3EpisodeSpan:
        self.load_episodes()
        assert self._episode_by_index is not None
        if episode_index not in self._episode_by_index:
            raise KeyError(f"Unknown episode_index={episode_index}")
        return self._episode_by_index[episode_index]

    def video_offset_s(self, *, episode_index: int, video_key: str) -> float:
        span = self.episode_span(episode_index)
        if video_key not in span.videos:
            raise KeyError(f"Episode {episode_index} missing span for {video_key!r}")
        return float(span.videos[video_key].from_timestamp_s)


