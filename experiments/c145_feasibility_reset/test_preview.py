"""Validate the new opponent's physical contract independently of success."""
import numpy as np
from experiments.c145_feasibility_reset.planar import Config,norm,wrap
from experiments.c145_feasibility_reset.preview_evader import PreviewWorld


def test_preview_turn_speed_and_position_contract_near_obstacles():
    w=PreviewWorld(range(1456000,1456032),Config(obstacles=4))
    for _ in range(300):
        heading=w.heading.copy();pos=w.target.copy()
        command=w.evader_command()
        assert np.max(np.abs(wrap(w.heading-heading)))<=.08+1e-10
        assert norm(command).max()<=3.2+1e-10
        np.testing.assert_array_equal(w.target,pos)
        # Exercise different positions/velocities without calling the opponent twice.
        w.tv+=.1*np.clip(command-w.tv,-2.,2.)
        w.target+=.1*w.tv


def test_preview_can_brake_and_does_not_promise_unconditional_safety():
    w=PreviewWorld([1],Config(obstacles=1))
    w.target[:]=0.;w.tv[:]=[3.2,0.];w.heading[:]=0.
    w.obstacles[:]=[4.8,0.];w.radii[:]=1.2
    w.p[:]=[[-7.,-4.],[-9.,0.],[-7.,4.]]
    command=w.evader_command()
    assert norm(command)[0]<3.2
    assert np.isfinite(command).all()
