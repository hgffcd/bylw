# -*- coding: utf-8 -*-
"""Versioned construction helpers for safety layers."""
from __future__ import annotations

from typing import Any

from safety.cbf_qp import CBFSafetyLayer
from safety.cbf_dynamic_v2 import DynamicCBFSafetyLayer


SAFETY_VERSIONS = ("legacy", "dynamic_v2")


def build_safety_layer(
    env_cfg: Any,
    version: str = "legacy",
    safety_params: dict[str, Any] | None = None,
):
    params = dict(safety_params or {})
    if version == "legacy":
        if params:
            raise ValueError(
                "legacy safety does not accept dynamic_v2 safety_params")
        return CBFSafetyLayer(
            dt=env_cfg.dt,
            a_max=env_cfg.a_max,
            w_max=env_cfg.w_max,
        )
    if version == "dynamic_v2":
        return DynamicCBFSafetyLayer.from_env_config(env_cfg, params)
    raise ValueError(
        f"unknown safety version {version!r}; expected one of {SAFETY_VERSIONS}")
