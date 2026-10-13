"""Same 16 development maps, isolated time-matching plus original baseline."""
from experiments.c229_fixed3_progress_bypass.design import groups as previous_groups
ARMS=[c+'_'+m for c in ['budget','progress','matched'] for m in ['predict','all']]
REUSED_ARMS=ARMS[:4];NEW_ARMS=ARMS[4:]
CONFIG={a:dict(recovery='budget_transit',search_mode=a.split('_')[1],control_variant='none' if a.startswith('budget') else a.split('_')[0]) for a in ARMS}
def groups():
    gg=previous_groups()
    for k,g in enumerate(gg):g['arm_order']=NEW_ARMS[k%2:]+NEW_ARMS[:k%2]
    return gg
