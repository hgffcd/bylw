"""First 16 observed C228 maps become a development regression cohort."""
from experiments.c228_fixed3_unseen.design import groups as parent_groups
ARMS=['budget_predict','budget_all','progress_predict','progress_all']
REUSED_ARMS=ARMS[:2];NEW_ARMS=ARMS[2:]
CONFIG={a:dict(recovery='budget_transit',search_mode=a.split('_')[1],control_variant='progress' if a.startswith('progress') else 'none') for a in ARMS}
def groups():
    gg=parent_groups()[:4]
    for k,g in enumerate(gg):g['arm_order']=NEW_ARMS[k%2:]+NEW_ARMS[:k%2]
    return gg
