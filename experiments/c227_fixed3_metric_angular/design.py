"""Fixed first 16 C224 development maps; angular distance scaling by search-mode factorial."""
from experiments.c224_fixed3_search_factorial.design import groups as old_groups
ARMS=['budget_predict','budget_all','metric_predict','metric_all']
NEW_ARMS=ARMS[2:];REUSED_ARMS=ARMS[:2]
CONFIG={a:dict(recovery='budget_transit',search_mode=a.split('_')[1],control_variant='metric' if a.startswith('metric') else 'none') for a in ARMS}
def groups():
    gg=old_groups()[:4]
    for k,g in enumerate(gg):g['arm_order']=NEW_ARMS[k%2:]+NEW_ARMS[:k%2]
    return gg
