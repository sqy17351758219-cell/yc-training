"""torch.distributed helpers for 8x141GB launches.

Every stage is launched with `torchrun --nproc_per_node 8`. Rank r processes
shard `items[r::world_size]`; results are gathered to rank 0 for writing. This
module is only imported inside the distributed scripts (it needs torch).
"""

from __future__ import annotations

import os
from typing import List, Sequence, TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def init_distributed(backend: str = "nccl") -> None:
    """Initialize the process group from torchrun env vars (idempotent)."""
    if dist.is_available() and not dist.is_initialized():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend=backend)
            torch.cuda.set_device(local_rank())


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main() -> bool:
    return rank() == 0


def device() -> "torch.device":
    """Per-rank device. Judges must live on cuda:{LOCAL_RANK} to avoid piling
    onto card 0 and OOMing (blueprint pitfall)."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank()}")
    return torch.device("cpu")


def barrier() -> None:
    if is_dist():
        dist.barrier()


def shard(items: Sequence[T]) -> List[T]:
    """This rank's slice: items[rank::world_size]."""
    return list(items[rank():: world_size()]) if is_dist() else list(items)


def gather_objects(local_list: List[T]) -> List[T]:
    """All-gather a list of picklable objects; returned flattened on every rank."""
    if not is_dist():
        return list(local_list)
    gathered: List[List[T]] = [None] * world_size()  # type: ignore
    dist.all_gather_object(gathered, local_list)
    out: List[T] = []
    for part in gathered:
        out.extend(part)
    return out


def cleanup() -> None:
    if is_dist():
        dist.destroy_process_group()
