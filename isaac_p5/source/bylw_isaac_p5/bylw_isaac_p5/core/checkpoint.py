"""CPU-only frozen actor checkpoint contract checks."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ActorCheckpointReport:
    checkpoint: str
    sha256: str
    actor_key_count: int
    input_dim: int
    action_dim: int
    hidden_dim: int
    log_std_dim: int

    def as_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load_tensor_only(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def inspect_actor_checkpoint(
    checkpoint: str | Path,
    expected_input_dim: int,
    expected_action_dim: int,
) -> ActorCheckpointReport:
    path = Path(checkpoint).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _torch_load_tensor_only(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("actor"), dict):
        raise ValueError("checkpoint must contain an actor state_dict")
    actor = payload["actor"]
    required = ("body.0.weight", "mu.weight", "log_std")
    missing = [name for name in required if name not in actor]
    if missing:
        raise ValueError(f"actor checkpoint missing keys: {missing}")
    input_dim = int(actor["body.0.weight"].shape[1])
    hidden_dim = int(actor["body.0.weight"].shape[0])
    action_dim = int(actor["mu.weight"].shape[0])
    log_std_dim = int(actor["log_std"].numel())
    if input_dim != expected_input_dim:
        raise ValueError(f"actor input dim {input_dim} != expected {expected_input_dim}")
    if action_dim != expected_action_dim or log_std_dim != expected_action_dim:
        raise ValueError(
            f"actor action dims mu={action_dim}, log_std={log_std_dim}, expected={expected_action_dim}"
        )
    return ActorCheckpointReport(
        checkpoint=str(path),
        sha256=sha256_file(path),
        actor_key_count=len(actor),
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        log_std_dim=log_std_dim,
    )
