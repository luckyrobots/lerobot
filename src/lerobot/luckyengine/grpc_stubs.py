from __future__ import annotations

import importlib
import os
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GeneratedStubs:
    tmp_dir: str
    pb2: Any
    pb2_grpc: Any


def _find_workspace_root(start: Path) -> Path:
    """
    Best-effort discovery of the monorepo root containing both `LuckyEngine/` and `lerobot/`.
    """
    cur = start.resolve()
    for p in (cur, *cur.parents):
        if (p / "LuckyEngine").is_dir() and (p / "lerobot").is_dir():
            return p
    # Fall back to cwd traversal.
    cur = Path.cwd().resolve()
    for p in (cur, *cur.parents):
        if (p / "LuckyEngine").is_dir() and (p / "lerobot").is_dir():
            return p
    raise FileNotFoundError("Could not locate workspace root containing both `LuckyEngine/` and `lerobot/`")


def default_proto_path() -> Path:
    root = _find_workspace_root(Path(__file__))
    return root / "LuckyEngine" / "Hazel-ScriptCore" / "Source" / "Hazel" / "Net" / "Grpc" / "Proto" / "hazel_rpc.proto"


@lru_cache(maxsize=8)
def generate_python_stubs(proto_path: str | os.PathLike[str]) -> GeneratedStubs:
    """
    Generate Python gRPC stubs from `hazel_rpc.proto` into a temp dir and import them.

    This mirrors `LuckyEngine/scripts/grpc_env.py` but is packaged for reuse in LeRobot tooling.
    """
    try:
        from grpc_tools import protoc  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing grpcio-tools. Install it (and grpcio) to use LuckyEngine gRPC:\n"
            "  python -m pip install grpcio grpcio-tools"
        ) from e

    proto_path = str(Path(proto_path).resolve())
    proto_dir = str(Path(proto_path).parent)

    tmp_dir = tempfile.mkdtemp(prefix="lerobot_lucky_grpc_py_")
    out_dir = tmp_dir

    # Make output importable.
    Path(out_dir, "__init__.py").write_text("# generated\n", encoding="utf-8")

    args = [
        "protoc",
        f"-I{proto_dir}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        os.path.basename(proto_path),
    ]

    cwd = os.getcwd()
    try:
        os.chdir(proto_dir)
        rc = protoc.main(args)
    finally:
        os.chdir(cwd)

    if rc != 0:  # pragma: no cover
        raise RuntimeError(f"protoc failed with exit code {rc}")

    # Ensure fresh import from the generated output directory.
    sys.path.insert(0, out_dir)
    pb2 = importlib.import_module("hazel_rpc_pb2")
    pb2_grpc = importlib.import_module("hazel_rpc_pb2_grpc")

    return GeneratedStubs(tmp_dir=tmp_dir, pb2=pb2, pb2_grpc=pb2_grpc)


