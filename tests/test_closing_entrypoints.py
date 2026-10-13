from argparse import Namespace
from pathlib import Path
import pytest
from reproduce.closing_native import build,cases

def args(chapter,case,arm,**kw):
    return Namespace(chapter=chapter,case=case,arm=arm,n=kw.get('n'),seed_base=None,
        duration=kw.get('duration'),isaac_python='isaac-python',gpu=2,out=Path('fresh-output'))

def test_final_modes_are_explicit():
    for arm in ['ordinary_predict','ordinary_single','ordinary_all','budget_all']:
        cmd,g=build(args('4','ch4_final3_g0',arm))
        assert cmd[cmd.index('--device')+1]=='cuda:0'
        assert cmd[cmd.index('--agents')+1]=='3'
        assert g['times']==[] and g['duration']==60

def test_withdrawal_reference_and_strategy():
    for arm in ['active','retained']:
        cmd,g=build(args('5','ch5_5to3_g0',arm))
        assert cmd[cmd.index('--failure-times')+1]=='2.5,3.0'
        assert cmd[cmd.index('--reconfig')+1]=={'active':'active_polar','retained':'retained_slots'}[arm]
        assert g['agents']==5

def test_invalid_contracts_fail_before_launch():
    with pytest.raises(ValueError):build(args('4','ch5_5to3_g0','ordinary_all'))
    with pytest.raises(ValueError):build(args('5','ch5_5to3_g0','active',duration=5))
    with pytest.raises(ValueError):build(args('4','ch4_final3_g0','ordinary_all',n=0))
    assert len(cases())==20
