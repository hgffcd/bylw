"""Fixed three agents; paired 2 x 2, no map generation or outcome screening."""
ARMS=['ordinary_single','ordinary_all','budget_single','budget_all']
CONFIG={a:dict(recovery='budget_transit' if a.startswith('budget') else 'routed_transit',
               search_mode='all' if a.endswith('_all') else 'single') for a in ARMS}

def groups():
    return [dict(label=f'fixed3_g{k}',seed=22440000+16*k,n=4,gpu=2+k%2,
                 agents=3,duration=60.,windows=[30,45,60],
                 arm_order=ARMS[k%4:]+ARMS[:k%4]) for k in range(8)]
