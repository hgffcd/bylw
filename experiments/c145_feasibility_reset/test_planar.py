"""Semantic checks for the new task, not regression tests for old Isaac."""
from dataclasses import replace
import numpy as np
from experiments.c145_feasibility_reset.planar import Config,World,geometry,visibility,run,norm


def test_geometry_requires_surround_and_safe_annulus():
    c=Config();t=np.zeros((1,2));a=np.arange(3)*2*np.pi/3
    p=4*np.stack([np.cos(a),np.sin(a)],-1)[None]
    assert geometry(p,t,c)[0].item()
    assert not geometry(p*.2,t,c)[0].item()
    assert not geometry(p*2,t,c)[0].item()
    p[0,2]=p[0,1]+.01
    assert not geometry(p,t,c)[0].item()


def test_blind_controller_cannot_read_new_target_truth():
    c=Config(blackout_start=0,blackout_duration=2)
    a,b=World([18,19],c),World([18,19],c)
    b.target+=100;b.tv+=10
    ao,bo=a.observe(),b.observe()
    for key in ao:np.testing.assert_array_equal(ao[key],bo[key])
    np.testing.assert_array_equal(a.nominal(ao),b.nominal(bo))
    np.testing.assert_array_equal(a.safety_filter(a.nominal(ao),ao),b.safety_filter(b.nominal(bo),bo))


def test_seeds_independent_of_batch_and_order():
    c=Config(obstacles=4,duration=1)
    a,_=run([19,18],c);b,_=run([18],c)
    assert a['episodes'][1]==b['episodes'][0]


def test_dynamics_speed_acceleration_and_no_teleport():
    w=World([18,19],Config(evader_ratio=1.2))
    for _ in range(10):
        p,v,tv=w.p.copy(),w.pv.copy(),w.tv.copy()
        w.step(w.observe())
        assert norm(w.pv).max()<=4+1e-9
        assert norm(w.tv).max()<=4.8+1e-9
        assert norm(w.pv-v).max()<=.4+1e-9
        assert norm(w.tv-tv).max()<=.4+1e-9
        assert norm(w.p-p).max()<=.4+1e-9


def test_first_hit_does_not_end_and_hold_is_contiguous():
    w=World([18],Config(duration=1,evader='cv',evader_ratio=0))
    a=np.arange(3)*2*np.pi/3
    w.target[:]=0;w.last[:]=0;w.p[:]=4.2*np.stack([np.cos(a),np.sin(a)],-1)
    for _ in range(10):w.step(w.observe())
    assert w.hit_geometry[0]==.1 and w.time==1.
    assert w.max_geometry[0]==10 and np.isnan(w.hold2_time[0])
    w.p[0,0]=[20,0];w.step(w.observe())
    assert w.run_geometry[0]==0


def test_blackout_exactly_twenty_steps_and_all_seeds_counted():
    c=Config(duration=4,blackout_start=1,blackout_duration=2)
    r,_=run([18,19,20],c)
    assert r['summary']['n']==3
    assert all(abs(e['team_loss_s']-2.)<1e-9 for e in r['episodes'])


def test_invalid_evader_episode_is_not_accepted_as_safe():
    w=World([18],Config(obstacles=1))
    w.obstacles[0,0]=w.target[0];w.step(w.observe())
    assert w.target_obstacle_invalid[0] and not w.records()[0]['safe_full_episode']


def test_fov_occlusion_and_team_measurement_distinct():
    w=World([18],Config(sensor='visual',obstacles=1))
    w.obstacles[0,0]=(w.p[0,0]+w.target[0])/2;w.radii[:]=.3
    obs=w.observe()
    assert not w.visible[0,0] and obs['measured'][0]
