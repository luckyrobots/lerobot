#!/usr/bin/env python

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import datasets
import packaging.version
import torch
import torch.utils
from datasets import load_dataset

from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.utils import (
    check_delta_timestamps,
    check_timestamps_sync,
    get_delta_indices,
    hf_transform_to_torch,
    load_info,
    load_stats,
)
from lerobot.datasets.video_utils import decode_video_frames, get_safe_default_codec

from .session_reader import LeRobotV3Session


V3_CODEBASE_VERSION = "v3.0"


def _is_mask_key(key: str) -> bool:
    return key.startswith("observation.masks.")


def _resolve_local_dataset_root(repo_id: str, root: str | Path | None) -> Path:
    """Resolve the on-disk dataset root.

    LeRobot usually uses `root/repo_id` as the dataset folder, but many local workflows pass the dataset
    folder directly as `root`. We accept both:
    - root/meta/info.json exists -> root is dataset folder
    - root/repo_id/meta/info.json exists -> root/repo_id is dataset folder
    """
    if root is None:
        return HF_LEROBOT_HOME / repo_id
    root = Path(root)
    if (root / "meta" / "info.json").is_file():
        return root
    if (root / repo_id / "meta" / "info.json").is_file():
        return root / repo_id
    return root


@dataclass
class LeRobotDatasetMetadataV3:
    repo_id: str
    root: str | Path
    revision: str | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.info = load_info(self.root)
        self._version = packaging.version.parse(self.info["codebase_version"])
        if self._version < packaging.version.parse(V3_CODEBASE_VERSION):
            raise ValueError(
                f"LeRobotDatasetMetadataV3 expects >= {V3_CODEBASE_VERSION}, got {self.info['codebase_version']!r}"
            )

        # Load stats (meta/stats.json) if present.
        self.stats = load_stats(self.root) or {}

        # Filter features: ignore masks by default.
        feats = dict(self.info["features"])
        feats = {k: v for k, v in feats.items() if not _is_mask_key(k)}
        self.info["features"] = feats

        # Tasks are in meta/tasks.parquet
        self._session = LeRobotV3Session(self.root)
        self.tasks = self._session.load_tasks()
        self.task_to_task_index = {task: i for i, task in self.tasks.items()}

        # Episodes are in meta/episodes parquet; parse only camera keys we keep.
        self._episodes_spans = self._session.load_episodes(include_video_keys=self.video_keys)

        # Build v2-like episodes dict used by some utilities.
        self.episodes: Dict[int, dict] = {}
        default_task = self.tasks.get(0, next(iter(self.tasks.values()), ""))
        for s in self._episodes_spans:
            self.episodes[s.episode_index] = {
                "episode_index": s.episode_index,
                "tasks": [default_task] if default_task else [],
                "length": s.length_frames,
            }

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    @property
    def features(self) -> dict[str, dict]:
        return self.info["features"]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def image_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def total_episodes(self) -> int:
        return int(self.info["total_episodes"])

    @property
    def total_frames(self) -> int:
        return int(self.info["total_frames"])

    def episode_span(self, episode_index: int):
        return self._session.episode_span(episode_index)

    def video_path_for(self, *, episode_index: int, video_key: str) -> Path:
        span = self.episode_span(episode_index)
        if video_key not in span.videos:
            raise KeyError(f"Episode {episode_index} has no video span for {video_key!r}")
        v = span.videos[video_key]
        return self._session.video_mp4_path(video_key=video_key, chunk_index=v.chunk_index, file_index=v.file_index)

    def video_offset_s(self, *, episode_index: int, video_key: str) -> float:
        return self._session.video_offset_s(episode_index=episode_index, video_key=video_key)


class LeRobotDatasetV3(torch.utils.data.Dataset):
    """LeRobot v3 session dataset.

    Implements the same high-level API as `LeRobotDataset`, but reads:
    - data shards from `data/chunk-*/file-*.parquet` (no per-row video references)
    - video frames using episode spans from `meta/episodes/*.parquet`

    By default, mask videos are excluded from `meta.features` so policies won't treat them as required inputs.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,  # unused; kept for parity with LeRobotDataset
        force_cache_sync: bool = False,  # unused; kept for parity with LeRobotDataset
        video_backend: str | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.root = _resolve_local_dataset_root(repo_id, root)
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision
        self.video_backend = video_backend if video_backend else get_safe_default_codec()

        # Metadata
        self.meta = LeRobotDatasetMetadataV3(repo_id=self.repo_id, root=self.root, revision=self.revision)

        # Load actual data (parquet shards)
        self.hf_dataset = self.load_hf_dataset()

        # If a subset of episodes is requested, filter rows.
        if self.episodes is not None:
            wanted = set(int(e) for e in self.episodes)
            # `filter` operates on Arrow and is OK for moderate dataset sizes.
            self.hf_dataset = self.hf_dataset.filter(lambda ex: int(ex["episode_index"]) in wanted)

        # Establish episode index mapping (for robust episode_data_index lookup).
        episode_indices = torch.stack(list(self.hf_dataset["episode_index"])).view(-1).numpy()
        unique_episode_indices: list[int] = []
        for e in episode_indices.tolist():
            if not unique_episode_indices or unique_episode_indices[-1] != int(e):
                unique_episode_indices.append(int(e))
        self._episode_index_to_local = {ep: i for i, ep in enumerate(unique_episode_indices)}

        # Build episode_data_index (exclusive 'to') for *selected* episodes in dataset order.
        lengths = []
        for ep in unique_episode_indices:
            span = self.meta.episodes.get(ep)
            if span is None:
                # Fallback: compute from data if meta is missing (should not happen).
                lengths.append(int((episode_indices == ep).sum()))
            else:
                lengths.append(int(span["length"]))
        cumulative = torch.tensor(lengths, dtype=torch.long).cumsum(0)
        from_ = torch.cat([torch.zeros(1, dtype=torch.long), cumulative[:-1]], dim=0)
        self.episode_data_index = {"from": from_, "to": cumulative}

        # Setup delta_indices
        self.delta_indices = None
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        # Check timestamps (episode-local timestamps are stored in the parquet)
        timestamps = torch.stack(list(self.hf_dataset["timestamp"])).view(-1).numpy()
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        # Use local episode indices (0..num_selected-1) for boundary masking.
        local_ep_indices = torch.tensor([self._episode_index_to_local[int(e)] for e in episode_indices], dtype=torch.long)
        check_timestamps_sync(timestamps, local_ep_indices.numpy(), ep_data_index_np, self.fps, self.tolerance_s)

    def load_hf_dataset(self) -> datasets.Dataset:
        path = str(self.root / "data")
        hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        return len(self.hf_dataset)

    @property
    def num_episodes(self) -> int:
        return len(self._episode_index_to_local)

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    def _get_query_indices(self, idx: int, ep_local: int) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        ep_start = int(self.episode_data_index["from"][ep_local].item())
        ep_end = int(self.episode_data_index["to"][ep_local].item())
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start) or (idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        *,
        current_ts: float,
        query_indices: dict[str, list[int]] | None,
    ) -> dict[str, list[float]]:
        query_timestamps: dict[str, list[float]] = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps_col = self.hf_dataset.select(query_indices[key])["timestamp"]
                # HF datasets returns a `Column` here; convert to a list of tensors/scalars before stacking.
                timestamps = list(timestamps_col)
                if len(timestamps) == 0:
                    query_timestamps[key] = []
                elif isinstance(timestamps[0], torch.Tensor):
                    query_timestamps[key] = torch.stack(timestamps).view(-1).tolist()
                else:
                    query_timestamps[key] = torch.tensor(timestamps).view(-1).tolist()
            else:
                query_timestamps[key] = [float(current_ts)]
        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        # Only query non-video keys; v3 data parquet does not contain per-row video metadata anyway.
        out: dict[str, torch.Tensor] = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            col = self.hf_dataset.select(q_idx)[key]
            values = list(col)  # HF returns `Column`; torch.stack can't consume it directly.
            if len(values) == 0:
                out[key] = torch.empty((0,))
                continue
            if isinstance(values[0], torch.Tensor):
                out[key] = torch.stack(values)
            else:
                out[key] = torch.tensor(values)
        return out

    def _query_videos(
        self,
        *,
        query_timestamps: dict[str, list[float]],
        episode_index: int,
    ) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {}
        for vid_key, ep_local_ts in query_timestamps.items():
            # Convert episode-local timestamps to shard-local timestamps.
            offset_s = self.meta.video_offset_s(episode_index=episode_index, video_key=vid_key)
            ts_in_video = [float(offset_s + float(t)) for t in ep_local_ts]
            video_path = self.meta.video_path_for(episode_index=episode_index, video_key=vid_key)
            frames = decode_video_frames(video_path, ts_in_video, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)
        return item

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
        item = self.hf_dataset[idx]
        episode_index = int(item["episode_index"].item())
        ep_local = self._episode_index_to_local[episode_index]

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_local)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        # Decode videos
        if len(self.meta.video_keys) > 0:
            current_ts = float(item["timestamp"].item())
            query_timestamps = self._get_query_timestamps(current_ts=current_ts, query_indices=query_indices)
            video_frames = self._query_videos(query_timestamps=query_timestamps, episode_index=episode_index)
            item = {**video_frames, **item}

        # Image transforms (torchvision v2) apply on camera tensors (C,H,W) or (T,C,H,W).
        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task string
        task_idx = int(item["task_index"].item())
        item["task"] = self.meta.tasks.get(task_idx, "")

        return item

    def __repr__(self) -> str:
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )


