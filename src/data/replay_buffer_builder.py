"""ReplayBuffer construction from canonical dataset dicts."""

from __future__ import annotations

from tianshou.data import ReplayBuffer

from src.core.types import DatasetDict
from src.data.schema import validate_and_standardize_dataset


def build_replay_buffer(
    data_dict: DatasetDict,
    buffer_size: int | None = None,
    *,
    validate: bool = True,
) -> ReplayBuffer:
    """Build a Tianshou ReplayBuffer from standardized dataset arrays."""

    data = validate_and_standardize_dataset(data_dict) if validate else data_dict
    n = data["obs"].shape[0]

    if buffer_size is not None and buffer_size < n:
        data = {k: v[:buffer_size] for k, v in data.items()}

    return ReplayBuffer.from_data(
        obs=data["obs"],
        act=data["act"],
        rew=data["rew"],
        done=data["done"],
        obs_next=data["obs_next"],
        terminated=data["terminated"],
        truncated=data["truncated"],
    )
