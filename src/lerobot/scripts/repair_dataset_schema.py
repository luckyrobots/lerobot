#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Repair a local LeRobot dataset that has nested observation struct and float64 lists.

This script rewrites every Parquet file under `data/chunk-*/file-*.parquet` so that:
- `observation.state` exists as a flat column (flattened from a nested `observation` struct)
- `action`, `observation.state`, and `timestamp` are cast to float32

Usage:

python -m lerobot.scripts.repair_dataset_schema \
  --dataset-root D:\\path\\to\\dataset\\session_YYYY-MM-DD_hh-mm-ss

Notes:
- Files are rewritten atomically via a temporary file placed next to the original, then replaced.
- A backup of each original Parquet is saved as `<file>.bak` unless `--no-backup` is set.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def cast_list_float(array: pa.ChunkedArray | pa.Array, dtype: pa.DataType) -> pa.ChunkedArray:
    target = pa.list_(dtype)
    if isinstance(array, pa.ChunkedArray):
        return pa.chunked_array([chunk.cast(target) for chunk in array.chunks])
    return pa.chunked_array([array.cast(target)])


def fix_table_schema(tbl: pa.Table) -> tuple[pa.Table, bool]:
    """Return a fixed table and whether a modification was applied."""
    modified = False

    cols: list[tuple[str, pa.ChunkedArray]] = []
    names = set(tbl.column_names)

    # Extract nested observation.state if present as a struct
    observation_state_col: pa.ChunkedArray | None = None
    if "observation" in names:
        obs = tbl.column("observation")
        if pa.types.is_struct(obs.type):
            try:
                # Merge chunks into a single StructArray for easier field access
                struct_arr = pa.chunked_array(obs.chunks).combine_chunks().chunk(0)
                field = struct_arr.field("state")
                observation_state_col = pa.chunked_array([field])
                modified = True
            except (KeyError, AttributeError):
                pass

    # Build new columns, flattening and casting where needed
    for name in tbl.column_names:
        if name == "observation":
            # Skip the nested struct column; we will write flattened key below if extracted
            continue

        col = tbl.column(name)

        if name == "action":
            # Ensure list<float32>
            cols.append((name, cast_list_float(col, pa.float32())))
            if not pa.types.is_list(col.type) or col.type.value_type != pa.float32():
                modified = True
            continue

        if name == "timestamp":
            # Scalars to float32
            if col.type != pa.float32():
                cols.append((name, col.cast(pa.float32())))
                modified = True
            else:
                cols.append((name, col))
            continue

        # Keep other columns unchanged
        cols.append((name, col))

    # Append flattened observation.state if available
    if observation_state_col is not None:
        # Cast to list<float32>
        cols.append(("observation.state", cast_list_float(observation_state_col, pa.float32())))

    # If observation.state already exists but wrong dtype, fix it
    if observation_state_col is None and "observation.state" in names:
        col = tbl.column("observation.state")
        if not (pa.types.is_list(col.type) and col.type.value_type == pa.float32()):
            cols = [(n, c) for (n, c) in cols if n != "observation.state"]
            cols.append(("observation.state", cast_list_float(col, pa.float32())))
            modified = True

    # Ensure deterministic column order
    desired_order = [
        "index",
        "episode_index",
        "frame_index",
        "timestamp",
        "task_index",
        "observation.state",
        "action",
    ]
    other = [n for (n, _) in cols if n not in desired_order]
    ordered_names = [n for n in desired_order if any(n == x for (x, _) in cols)] + other
    name_to_col = {n: c for (n, c) in cols}
    new_cols = [(n, name_to_col[n]) for n in ordered_names]

    new_tbl = pa.table({n: c for (n, c) in new_cols})
    return new_tbl, modified


def rewrite_parquet_file(fpath: Path, *, backup: bool = True) -> bool:
    tbl = pq.read_table(fpath)
    new_tbl, modified = fix_table_schema(tbl)
    if not modified:
        return False

    tmp_path = fpath.with_suffix(fpath.suffix + ".tmp")
    pq.write_table(new_tbl, tmp_path)

    if backup:
        bak = fpath.with_suffix(fpath.suffix + ".bak")
        if bak.exists():
            bak.unlink()
        shutil.move(str(fpath), str(bak))
    shutil.move(str(tmp_path), str(fpath))
    return True


def repair_dataset(root: Path, *, backup: bool = True) -> tuple[int, int]:
    data_dir = root / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"'data' directory not found under {root}")

    files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    total = len(files)
    fixed = 0
    for f in files:
        try:
            if rewrite_parquet_file(f, backup=backup):
                fixed += 1
                logging.info(f"Repaired {f}")
        except Exception as e:
            logging.error(f"Failed to repair {f}: {e}")
            raise
    return fixed, total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, required=True, help="Path to the dataset folder containing meta/data/videos")
    p.add_argument("--no-backup", action="store_true", help="Do not create .bak backups of original Parquet files")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    root = Path(args.dataset_root)
    fixed, total = repair_dataset(root, backup=not args.no_backup)
    logging.info(f"Repaired {fixed}/{total} parquet files under {root / 'data'}")


if __name__ == "__main__":
    main()


