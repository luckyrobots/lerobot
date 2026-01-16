import pytest


def test_v3_metadata_filters_masks():
    # Local integration test: the repo vendors a v3 session under `lerobot/dataset/`.
    from lerobot.datasets.v3.lerobot_dataset_v3 import LeRobotDatasetMetadataV3

    root = "lerobot/dataset/session_20260109_145524"
    meta = LeRobotDatasetMetadataV3(repo_id="session_20260109_145524", root=root)

    assert "observation.masks.CameraLeft" not in meta.features
    assert "observation.masks.CameraSide" not in meta.features

    # Only the two RGB views should be treated as cameras by default.
    assert "observation.images.CameraLeft" in meta.camera_keys
    assert "observation.images.CameraSide" in meta.camera_keys
    assert all(not k.startswith("observation.masks.") for k in meta.camera_keys)


def test_factory_uses_v3_dataset_class():
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy_config

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id="session_20260109_145524",
            # Point directly to the v3 session folder.
            root="lerobot/dataset/session_20260109_145524",
            episodes=None,
        ),
        # Any policy config works here: dataset creation only needs delta index info.
        policy=make_policy_config("diffusion", push_to_hub=False),
    )

    ds = make_dataset(cfg)
    assert ds.__class__.__name__ == "LeRobotDatasetV3"


