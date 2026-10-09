"""Explicit exclusions for three archival checks whose inputs are not in the code-only backup."""
from pathlib import Path
import pytest

ROOT = Path(__file__).parent
ARCHIVAL = {
    'test_c19_video_replays_current_task_and_uses_unscaled_closeup_camera':
        ('scripts/record_p5_2_native_success.py', 'Legacy video recorder is outside the curated runtime export'),
    'test_p5_1r_source_keeps_legacy_default_and_dense_physical_telemetry':
        ('scripts/run_p5_1_zero_shot.py', 'Legacy zero-shot orchestration is outside the curated runtime export'),
    'test_refinement_finds_routed_legal_slots_in_tight_regression_observation':
        ('experiments/c186_precision_recovery/tight_observation.json', 'Historical observation data is intentionally not uploaded'),
}


def pytest_collection_modifyitems(items):
    for item in items:
        if item.name in ARCHIVAL:
            relative, reason = ARCHIVAL[item.name]
            if not (ROOT / relative).is_file():
                item.add_marker(pytest.mark.skip(reason=reason))
