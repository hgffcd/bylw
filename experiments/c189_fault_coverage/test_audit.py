import numpy as np
from experiments.c189_fault_coverage.audit import temporal_coverage


def test_loss_between_two_withdrawals_is_not_a_postfinal_loss():
    observed = np.ones(100, bool); observed[20:29] = False
    result = temporal_coverage(observed, np.ones(100, bool), False, [25, 30])
    assert result['lost_at_withdrawals'] == [True, False]
    assert result['loss_after_first_withdrawal']
    assert not result['loss_after_last_withdrawal']
    assert result['safe_capture_after_first_last_reacquisition']


def test_capture_before_last_reacquisition_is_not_recovery_success():
    observed = np.ones(100, bool); observed[80:90] = False
    result = temporal_coverage(observed, np.ones(100, bool), False, [10])
    assert result['reacquired_after_last_last_loss']
    assert not result['safe_capture_after_last_last_reacquisition']
    assert result['hold_after_last_last_reacquisition_s'] == 1.


def test_unrecovered_final_loss_and_any_unsafe_episode_are_not_success():
    observed = np.ones(100, bool); observed[20:] = False
    result = temporal_coverage(observed, np.ones(100, bool), False, [10])
    assert not result['reacquired_after_last_last_loss']
    observed[30:] = True
    result = temporal_coverage(observed, np.ones(100, bool), True, [10])
    assert result['reacquired_after_last_last_loss']
    assert not result['safe_capture_after_last_last_reacquisition']
