# -*- coding: utf-8 -*-
"""Versioned predictor checkpoint I/O with legacy GRU compatibility."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

from belief.gru_predictor import GRUGaussianPredictor
from belief.transformer_predictor import TransformerGaussianPredictor


CHECKPOINT_FORMAT_VERSION = 2


def predictor_type(model: nn.Module) -> str:
    if isinstance(model, GRUGaussianPredictor):
        return "gru"
    if isinstance(model, TransformerGaussianPredictor):
        return "transformer"
    raise TypeError(f"unsupported predictor type: {type(model).__name__}")


def predictor_kwargs(model: nn.Module) -> dict:
    kind = predictor_type(model)
    if kind == "transformer":
        return model.model_kwargs()
    return {
        "in_dim": model.gru.input_size,
        "hidden": model.gru.hidden_size,
        "horizon": model.horizon,
        "rollout_eta": model.rollout_eta,
        "rollout_q0": model.rollout_q0,
    }


def checkpoint_payload(
    model: nn.Module,
    *,
    temperature: float | None = None,
    metadata: dict | None = None,
) -> dict:
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_type": predictor_type(model),
        "model_kwargs": predictor_kwargs(model),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    return payload


def save_predictor_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    temperature: float | None = None,
    metadata: dict | None = None,
) -> None:
    """Save a self-describing v2 checkpoint."""
    torch.save(
        checkpoint_payload(
            model, temperature=temperature, metadata=metadata),
        path,
    )


def _build_model(model_type: str, kwargs: dict) -> nn.Module:
    if model_type == "gru":
        return GRUGaussianPredictor(**kwargs)
    if model_type == "transformer":
        return TransformerGaussianPredictor(**kwargs)
    raise ValueError(f"unknown predictor model_type: {model_type!r}")


def load_predictor_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    freeze: bool = False,
) -> tuple[nn.Module, float | None, dict]:
    """Load v2 checkpoints and historical raw GRU state dictionaries.

    Returns ``(model, temperature, metadata)``. Historical checkpoints are
    identified by GRU parameter names and get ``metadata['legacy'] = True``.
    """
    loaded = torch.load(path, map_location=map_location)
    if not isinstance(loaded, Mapping):
        raise TypeError("predictor checkpoint must contain a mapping")

    if "state_dict" in loaded and "model_type" in loaded:
        model_type = str(loaded["model_type"])
        kwargs = dict(loaded.get("model_kwargs") or {})
        model = _build_model(model_type, kwargs)
        model.load_state_dict(loaded["state_dict"])
        temperature = loaded.get("temperature")
        metadata = dict(loaded.get("metadata") or {})
        metadata.update({
            "format_version": int(
                loaded.get("format_version", CHECKPOINT_FORMAT_VERSION)),
            "model_type": model_type,
            "model_kwargs": kwargs,
            "legacy": False,
        })
    else:
        state_dict = dict(loaded)
        temperature = state_dict.pop("temperature", None)
        required = {
            "gru.weight_ih_l0", "gru.weight_hh_l0", "head_mu.weight"}
        if not required.issubset(state_dict):
            missing = sorted(required.difference(state_dict))
            raise ValueError(
                "unrecognized legacy predictor checkpoint; "
                f"missing keys: {missing}")
        kwargs = {
            "in_dim": state_dict["gru.weight_ih_l0"].shape[1],
            "hidden": state_dict["gru.weight_hh_l0"].shape[1],
            "horizon": state_dict["head_mu.weight"].shape[0] // 2,
        }
        model = GRUGaussianPredictor(**kwargs)
        model.load_state_dict(state_dict)
        metadata = {
            "format_version": 1,
            "model_type": "gru",
            "model_kwargs": kwargs,
            "legacy": True,
        }

    model.eval()
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return (
        model,
        float(temperature) if temperature is not None else None,
        metadata,
    )
