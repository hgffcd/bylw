"""Read-only command-time path intervention and guard diagnostics."""
from experiments.c227_fixed3_metric_angular.telemetry import snapshot as old_snapshot,reset_snapshot

def snapshot(env):
    values=old_snapshot(env)
    for key in ['applied','on','stall','base','proposal']:
        values['cmd_progress_'+key]=getattr(env,'last_progress_'+key).copy()
    values['cmd_raw_command']=env.last_raw_command.copy()
    values['cmd_guard_delta']=env.last_guard_delta.copy()
    return values
