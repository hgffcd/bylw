"""Same 32 C224 maps; two new no-active-search arms, four frozen comparators."""
from experiments.c224_fixed3_search_factorial.design import groups as old_groups
CORES=['ordinary','budget']
MODES=['predict','single','all']
ARMS=[c+'_'+m for c in CORES for m in MODES]
NEW_ARMS=[c+'_predict' for c in CORES]
REUSED_ARMS=[a for a in ARMS if a not in NEW_ARMS]
CONFIG={a:dict(recovery='budget_transit' if a.startswith('budget') else 'routed_transit',search_mode=a.split('_')[1]) for a in ARMS}
def groups():
    gg=old_groups()
    for k,g in enumerate(gg):g['arm_order']=NEW_ARMS[k%2:]+NEW_ARMS[:k%2]
    return gg
