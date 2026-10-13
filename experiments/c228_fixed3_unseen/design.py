"""Prospective 32-map replication, frozen C227 control and sensor parameters."""
from experiments.c227_fixed3_metric_angular.design import ARMS,CONFIG

def groups():
    return [dict(label=f'unseen3_g{k}',seed=22840000+16*k,n=4,gpu=2+k%2,agents=3,
                 duration=60.,windows=[30,45,60],arm_order=ARMS[k%4:]+ARMS[:k%4]) for k in range(8)]
