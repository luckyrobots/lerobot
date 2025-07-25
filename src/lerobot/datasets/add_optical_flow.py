from __future__ import annotations

"""Add optical-flow features to an existing LeRobotDataset.

This command-line tool iterates over a *v2.x* LeRobotDataset, computes dense
optical-flow maps for specified camera image streams using the
`OpticalFlowProcessor`, and persists the results as new dataset features.

The implementation is intentionally conservative:
• It never modifies the original dataset in-place.  Instead, it creates a new
  dataset clone under `--output_root` with an updated metadata schema and new
  parquet files containing the flow tensors.
• All metadata validation / feature checks rely on existing helpers from
  `lerobot.datasets` to ensure consistency.
• Computation is performed episode-by-episode and frame-by-frame with
  incremental progress logging and graceful interruption handling (SIGINT).

Example
-------
$ python -m lerobot.datasets.add_optical_flow \
    --repo_id my_dataset \
    --input_root /datasets/original \
    --output_root /datasets/with_flow \
    --cameras observation.image_cam1 observation.image_cam2 \
    --device cuda
"""

from pathlib import Path
import argparse
import logging
import signal
import sys
from typing import List, Dict, Any

import numpy as np
import torch

from lerobot.utils.optical_flow_neuflow import OpticalFlowProcessor
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import FeatureType, aggregate_stats, write_json, STATS_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add optical-flow features to a LeRobotDataset")
    p.add_argument("--repo_id", required=True, help="Dataset repo-id (same as original)")
    p.add_argument("--input_root", type=Path, required=True, help="Root folder of _source_ dataset")
    p.add_argument("--output_root", type=Path, required=True, help="Where to write the new dataset with flow")
    p.add_argument(
        "--cameras",
        nargs="+",
        required=True,
        help="Camera keys to process (e.g. observation.image_cam1 observation.image_cam2)",
    )
    p.add_argument("--flow_suffix", default="flow", help="Suffix appended to camera key (default: 'flow')")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    p.add_argument("--batch", type=int, default=8, help="Batch size when computing flow (frames per call)")
    p.add_argument("--iters_s16", type=int, default=1, help="NeuFlow iterations @1/16")
    p.add_argument("--iters_s8", type=int, default=8, help="NeuFlow iterations @1/8")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output_root if it exists")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Core processing logic
# -----------------------------------------------------------------------------

def build_flow_feature_schema(example_img: torch.Tensor) -> Dict[str, Dict[str, Any]]:
    """Return a feature dict entry describing the flow tensor shape/dtype."""
    _, h, w = example_img.shape
    return {
        "dtype": "float32",
        "shape": (h, w, 2),  # H x W x UV
        "names": ["height", "width", "channels"],
    }


def clone_dataset(src_meta: LeRobotDatasetMetadata, dst_root: Path) -> LeRobotDatasetMetadata:
    """Create an empty clone of the source dataset metadata under dst_root."""
    if dst_root.exists():
        raise FileExistsError(f"Destination dataset path '{dst_root}' already exists. Use --overwrite to replace.")

    dst_root.mkdir(parents=True, exist_ok=False)

    # Copy meta directory (info/stats/tasks/episodes) first; actual parquet/videos copied lazily
    for meta_file in (src_meta.root / "meta").glob("*.*"):
        (dst_root / "meta").mkdir(parents=True, exist_ok=True)
        (dst_root / meta_file.relative_to(src_meta.root)).write_bytes(meta_file.read_bytes())

    # Load metadata from new location (will validate files later)
    return LeRobotDatasetMetadata(src_meta.repo_id, root=dst_root)


def main() -> None:  # noqa: C901 (complexity acceptable here)
    args = parse_args()

    # ------------------------------------------------------------------
    # Load source dataset & basic sanity checks
    # ------------------------------------------------------------------
    src_dataset = LeRobotDataset(args.repo_id, root=args.input_root)
    src_meta = src_dataset.meta

    missing_cams = [cam for cam in args.cameras if cam not in src_meta.camera_keys]
    if missing_cams:
        logger.error("Camera keys not found in dataset: %s", missing_cams)
        sys.exit(1)

    if args.output_root.exists() and not args.overwrite:
        logger.error("Output directory %s already exists. Use --overwrite to replace.", args.output_root)
        sys.exit(1)
    if args.overwrite and args.output_root.exists():
        import shutil

        shutil.rmtree(args.output_root)

    # ------------------------------------------------------------------
    # Prepare destination dataset
    # ------------------------------------------------------------------
    dst_meta = clone_dataset(src_meta, args.output_root)

    # Update metadata features
    example_idx = 0
    example = src_dataset[example_idx]
    for cam_key in args.cameras:
        flow_key = f"{cam_key}_{args.flow_suffix}"
        if flow_key in dst_meta.features:
            logger.warning("Flow feature '%s' already exists in destination dataset; skipping schema update", flow_key)
            continue
        schema = build_flow_feature_schema(example[cam_key])
        dst_meta.info["features"][flow_key] = schema
        # Add to camera_keys for future
        if schema["dtype"] in ["image", "video", "float32"]:
            pass  # nothing else required – camera_keys are computed property
    # Persist updated info.json
    write_json(dst_meta.info, dst_meta.root / "meta/info.json")

    # ------------------------------------------------------------------
    # Optical-flow processor initialisation
    # ------------------------------------------------------------------
    of_proc = OpticalFlowProcessor(device=args.device, iters_s16=args.iters_s16, iters_s8=args.iters_s8)

    # Graceful CTRL-C handling
    interrupted = False

    def _sigint_handler(signum, frame):  # noqa: D401
        nonlocal interrupted
        interrupted = True
        logger.warning("Interrupted – finishing current episode then exiting …")

    signal.signal(signal.SIGINT, _sigint_handler)

    # ------------------------------------------------------------------
    # Iterate episodes and augment data
    # ------------------------------------------------------------------
    for ep_idx in range(src_meta.total_episodes):
        ep_indices = np.where(src_dataset.hf_dataset["episode_index"] == ep_idx)[0]
        if len(ep_indices) == 0:
            continue

        logger.info("Processing episode %d (%d frames)", ep_idx, len(ep_indices))
        # Retrieve timestamps to preserve ordering
        ep_frames = src_dataset.hf_dataset.select(ep_indices)

        # Build new tensors for each camera flow feature
        flow_buffers: Dict[str, List[torch.Tensor]] = {f"{cam}_{args.flow_suffix}": [] for cam in args.cameras}

        prev_imgs: Dict[str, torch.Tensor] | None = None
        # Iterate frames sequentially (not batched for simplicity)
        for frame in ep_frames:
            if interrupted:
                break
            if prev_imgs is None:
                prev_imgs = {cam: frame[cam] for cam in args.cameras}
                # Zero-flow for first frame
                for cam in args.cameras:
                    h, w = frame[cam].shape[-2:]
                    flow_zero = torch.zeros(h, w, 2, dtype=torch.float32)
                    flow_buffers[f"{cam}_{args.flow_suffix}"].append(flow_zero)
                continue

            curr_imgs = {cam: frame[cam] for cam in args.cameras}
            for cam in args.cameras:
                flow_uv = of_proc.compute_flow(prev_imgs[cam], curr_imgs[cam], output_numpy=False)
                flow_buffers[f"{cam}_{args.flow_suffix}"].append(flow_uv.cpu())
            prev_imgs = curr_imgs

        if interrupted:
            break

        # ------------------------------------------------------------------
        # Write episode parquet with new flow features
        # ------------------------------------------------------------------
        from datasets import Dataset  # local import to avoid heavy deps if not used elsewhere
        from lerobot.datasets.utils import get_hf_features_from_features, embed_images, hf_transform_to_torch
        from lerobot.datasets.compute_stats import compute_episode_stats

        ep_len = len(ep_frames)

        # Build dict of existing columns
        ep_dict: Dict[str, list] = {}
        for col in ep_frames.column_names:
            ep_dict[col] = ep_frames[col]

        # Append new flow columns (convert tensors ➜ numpy)
        for flow_key, buffer in flow_buffers.items():
            ep_dict[flow_key] = [t.numpy() for t in buffer]

        # Get HuggingFace feature schema including new flow keys
        features_schema = get_hf_features_from_features(dst_meta.features)

        ep_dataset = Dataset.from_dict(ep_dict, features=features_schema, split="train")
        # Ensure image columns embedded as bytes (reuses existing util)
        ep_dataset = embed_images(ep_dataset)

        # Save to parquet under destination path
        ep_data_path = dst_meta.root / dst_meta.get_data_file_path(ep_idx)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)

        # ------------------------------------------------------------------
        # Update metadata: episodes list, global stats, etc.
        # ------------------------------------------------------------------
        episode_tasks = list(set(ep_frames["task"])) if "task" in ep_frames.column_names else []
        ep_stats = compute_episode_stats({k: np.stack(v) if isinstance(v[0], np.ndarray) else v for k, v in ep_dict.items()}, dst_meta.features)

        dst_meta.save_episode(ep_idx, ep_len, episode_tasks, ep_stats)

        logger.info("Episode %d saved (%d frames, %d new flow keys)", ep_idx, ep_len, len(flow_buffers))

    # All episodes processed – final stats already updated via save_episode
    write_json(dst_meta.stats, dst_meta.root / STATS_PATH)

    if interrupted:
        logger.warning("Processing interrupted by user; dataset might be incomplete.")
    else:
        logger.info("Flow augmentation completed successfully.")


if __name__ == "__main__":
    main() 