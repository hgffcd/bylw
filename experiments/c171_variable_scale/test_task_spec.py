import sys
from pathlib import Path

import pytest


ISAAC_SRC = Path(__file__).resolve().parents[2] / "isaac_p5" / "source" / "bylw_isaac_p5"
if str(ISAAC_SRC) not in sys.path:
    sys.path.insert(0, str(ISAAC_SRC))

from bylw_isaac_p5.core.config import P5TaskSpec  # noqa: E402


def test_default_contract_remains_three_pursuers():
    spec = P5TaskSpec()
    spec.validate()
    assert spec.n_pursuers == 3
    assert spec.min_capture_agents == 3


def test_variable_scale_opt_in_accepts_four_and_three_active_after_failure():
    spec = P5TaskSpec(n_pursuers=4, min_capture_agents=3, allow_variable_scale=True)
    spec.validate()
    assert spec.observation_dim > P5TaskSpec().observation_dim


def test_variable_scale_still_rejects_two_active_capture():
    with pytest.raises(ValueError):
        P5TaskSpec(n_pursuers=4, min_capture_agents=2, allow_variable_scale=True).validate()


def test_unopted_four_pursuer_contract_is_rejected():
    with pytest.raises(ValueError):
        P5TaskSpec(n_pursuers=4, min_capture_agents=4).validate()
