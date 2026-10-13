"""Frozen system prospective validation; no new-map generation at import."""
ARMS=['ordinary_predict','ordinary_single','ordinary_all','budget_all']
CONFIG={a:dict(recovery='budget_transit' if a.startswith('budget') else 'routed_transit',search_mode=a.split('_')[1]) for a in ARMS}
def groups():
    return [dict(label=f'final3_g{k}',seed=23340000+16*k,n=4,gpu=2+k%2,agents=3,
        duration=60.,windows=[30,45,60],arm_order=ARMS[k%4:]+ARMS[:k%4]) for k in range(8)]
