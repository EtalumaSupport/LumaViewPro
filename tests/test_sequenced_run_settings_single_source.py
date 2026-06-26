"""Guard the single source for settings-derived sequenced-run params.

Regression for the bench-found defect where the GUI scan path
(ui/protocol_settings.py) called SequencedCaptureRunner.run() without
forwarding keep_led_between_steps, so the across-move LED hold silently never
applied from the GUI even though the lifecycle test passed (it called run()
with the kwarg directly, like the API path). The run-invocation sites had
drifted: each forwarded its own hand-picked subset of the settings-derived
params, so a key present on one path but missing on another changed hardware
behavior on only that path.

config_helpers.get_sequenced_run_settings() is the single source the normal
acquisition paths spread into run(). These tests pin the helper contract and
inspect EVERY acquisition run() call site (identified by its run_mode= kwarg)
across modules/ and ui/, so a newly-added un-migrated site fails here rather
than shipping the bug. The autofocus-scan path is intentionally exempt: it must
NOT hold the LED across focus moves (photobleaching), so it does not spread the
helper and is identified by its SINGLE_AUTOFOCUS_SCAN run_mode.
"""

import pathlib
import re

import modules.config_helpers as config_helpers

_REPO = pathlib.Path(__file__).resolve().parents[1]
_OWNED_PARAMS = ('keep_led_between_steps', 'video_as_frames', 'separate_folder_per_channel')

_RUN_CALL = re.compile(r'\.run\(')
_SPREAD = '**config_helpers.get_sequenced_run_settings('


def _balanced_block(src: str, open_paren_idx: int) -> str:
    """Return src[start..close] for the call whose '(' is at open_paren_idx."""
    depth = 0
    for j in range(open_paren_idx, len(src)):
        if src[j] == '(':
            depth += 1
        elif src[j] == ')':
            depth -= 1
            if depth == 0:
                return src[open_paren_idx : j + 1]
    raise AssertionError('unbalanced parens while extracting run() call block')


def _acquisition_run_call_blocks():
    """Yield (path, block) for every SequencedCaptureRunner.run() acquisition
    call across modules/ and ui/. An acquisition call is any `.run(` whose arg
    block carries a run_mode= kwarg (the capture runner's required selector)."""
    for sub in ('modules', 'ui'):
        for path in sorted((_REPO / sub).glob('*.py')):
            src = path.read_text()
            for m in _RUN_CALL.finditer(src):
                block = _balanced_block(src, m.end() - 1)
                if 'run_mode=' in block:
                    yield path.relative_to(_REPO).as_posix(), block


def test_helper_passes_settings_values_through():
    out = config_helpers.get_sequenced_run_settings(
        {
            'keep_led_between_steps': True,
            'video_as_frames': True,
            'separate_folder_per_channel': True,
        }
    )
    assert out == {
        'keep_led_between_steps': True,
        'video_as_frames': True,
        'separate_folder_per_channel': True,
    }


def test_helper_defaults_missing_keys_off():
    assert config_helpers.get_sequenced_run_settings({}) == {
        'keep_led_between_steps': False,
        'video_as_frames': False,
        'separate_folder_per_channel': False,
    }


def test_helper_owns_exactly_the_settings_derived_run_params():
    assert set(config_helpers.get_sequenced_run_settings({})) == set(_OWNED_PARAMS)


def test_every_acquisition_run_call_is_classified():
    # Sanity: the scanner finds the known sites so the guards below are not
    # vacuously passing. >= 4 (API scan, GUI scan, GUI z-stack, AF scan).
    blocks = list(_acquisition_run_call_blocks())
    assert len(blocks) >= 4, f'expected >=4 acquisition run() sites, found {len(blocks)}'


def test_non_af_acquisition_sites_spread_the_helper():
    for rel, block in _acquisition_run_call_blocks():
        if 'SINGLE_AUTOFOCUS_SCAN' in block:
            continue  # AF scan is exempt by design (no LED hold across focus moves)
        assert _SPREAD in block, (
            f'{rel}: an acquisition run() call does not spread '
            f'get_sequenced_run_settings(); a settings-derived run param can '
            f'drift on this path (this is how the GUI scan/z-stack bug shipped).'
        )


def test_keep_led_between_steps_never_hand_passed_to_run():
    # The regressed param must arrive only via the helper spread, never as a
    # per-site kwarg on a run() call -- inline or multiline, AF path included.
    for rel, block in _acquisition_run_call_blocks():
        assert not re.search(r'keep_led_between_steps\s*=', block), (
            f'{rel}: a run() call hand-passes keep_led_between_steps= instead of '
            f'sourcing it from get_sequenced_run_settings(); re-divergence risk.'
        )
