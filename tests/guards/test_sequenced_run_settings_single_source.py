"""Guard the single source for settings-derived sequenced-run params.

Regression for the bench-found defect where the GUI scan path
(ui/protocol_settings.py) started the sequenced capture runner without
forwarding keep_led_between_steps, so the across-move LED hold silently never
applied from the GUI even though the lifecycle test passed (it passed the
kwarg directly, like the API path). The run-invocation sites had
drifted: each forwarded its own hand-picked subset of the settings-derived
params, so a key present on one path but missing on another changed hardware
behavior on only that path.

config_helpers.get_sequenced_run_settings() is the single source the
acquisition paths spread into SequencedCaptureRunner.prepare(). These tests
pin the helper contract and inspect EVERY acquisition prepare() call site
(identified by its run_mode= kwarg) across modules/ and ui/, so a
newly-added un-migrated site fails here rather than shipping the bug. The
autofocus-scan path is not exempt: its LED safety (it must NOT hold the LED
across focus moves, which would photobleach the sample) and its
no-artifacts folder layout are expressed once, inside the helper's
run_mode branch, so an AF scan spreads the helper like every other site and
still gets keep_led_between_steps=False and
separate_folder_per_channel=False whatever the user settings hold.
"""

import re

import pytest

import modules.config_helpers as config_helpers
from modules.image_mode import (
    IMAGE_MODE_8BIT,
    IMAGE_MODE_12BIT_SCIENTIFIC,
    OUTPUT_FORMAT_HYPERSTACK,
    OUTPUT_FORMAT_OME_TIFF,
    ImageCaptureConfig,
)
from modules.protocol_state_machine import SequencedCaptureRunMode
from tests.ast_seams import REPO_ROOT

_OWNED_PARAMS = (
    'image_capture_config',
    'keep_led_between_steps',
    'video_as_frames',
    'separate_folder_per_channel',
    'bf_af_for_fluorescence',
    'timestamp_overlay',
    'video_max_fps',
)

_RUN_CALL = re.compile(r'\.prepare\(')
_HELPER_CALL = 'config_helpers.get_sequenced_run_settings('
_SPREAD = '**' + _HELPER_CALL


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
    raise AssertionError('unbalanced parens while extracting prepare() call block')


def _acquisition_run_call_blocks():
    """Yield (path, block) for every SequencedCaptureRunner.prepare()
    acquisition call across modules/ and ui/. An acquisition call is any
    `.prepare(` whose arg block carries a run_mode= kwarg (the capture
    runner's required selector)."""
    for sub in ('modules', 'ui'):
        for path in sorted((REPO_ROOT / sub).glob('*.py')):
            src = path.read_text()
            for m in _RUN_CALL.finditer(src):
                block = _balanced_block(src, m.end() - 1)
                if 'run_mode=' in block:
                    yield path.relative_to(REPO_ROOT).as_posix(), block


def test_helper_passes_settings_values_through():
    out = config_helpers.get_sequenced_run_settings(
        {
            'keep_led_between_steps': True,
            'video_as_frames': True,
            'separate_folder_per_channel': True,
            'protocol': {'bf_af_for_fluorescence': True},
            'video': {'timestamp_overlay': False, 'max_fps': 5},
        },
        run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
    )
    assert out == {
        'image_capture_config': ImageCaptureConfig.from_image_mode(IMAGE_MODE_8BIT),
        'keep_led_between_steps': True,
        'video_as_frames': True,
        'separate_folder_per_channel': True,
        'bf_af_for_fluorescence': True,
        'timestamp_overlay': False,
        'video_max_fps': 5,
    }


def test_helper_defaults_missing_keys_to_the_shipped_defaults():
    # Two of the seven read a NESTED path (protocol.bf_af_for_fluorescence,
    # video.timestamp_overlay, video.max_fps) under a FLAT run-param name,
    # and two of the defaults are not False -- the overlay ships on and the
    # rate cap is a number, so "missing key" is not "everything off".
    assert config_helpers.get_sequenced_run_settings(
        {}, run_mode=SequencedCaptureRunMode.FULL_PROTOCOL
    ) == {
        'image_capture_config': ImageCaptureConfig.from_image_mode(IMAGE_MODE_8BIT),
        'keep_led_between_steps': False,
        'video_as_frames': False,
        'separate_folder_per_channel': False,
        'bf_af_for_fluorescence': False,
        'timestamp_overlay': True,
        'video_max_fps': 0,
    }


def test_helper_owns_exactly_the_settings_derived_run_params():
    assert set(
        config_helpers.get_sequenced_run_settings(
            {}, run_mode=SequencedCaptureRunMode.FULL_PROTOCOL
        )
    ) == set(_OWNED_PARAMS)


_AUTOFOCUS_RUN_MODES = (
    SequencedCaptureRunMode.SINGLE_AUTOFOCUS,
    SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN,
)


@pytest.mark.parametrize('run_mode', _AUTOFOCUS_RUN_MODES, ids=lambda m: m.value)
def test_autofocus_scan_forces_led_and_folder_safety(run_mode):
    """An autofocus run -- one position or every step -- must never hold
    the LED across focus moves (photobleaching) and must never split output
    per channel, whatever the user settings say. The guarantee lives in the
    helper, so it reaches both AF starters from one place instead of a
    literal at each."""
    settings = {
        'keep_led_between_steps': True,
        'video_as_frames': True,
        'separate_folder_per_channel': True,
    }
    out = config_helpers.get_sequenced_run_settings(settings, run_mode=run_mode)
    assert out['keep_led_between_steps'] is False, (
        'an autofocus scan must not hold the LED across focus moves even when '
        'the user has keep_led_between_steps on'
    )
    assert out['separate_folder_per_channel'] is False, (
        'an autofocus scan must not allocate per-channel folders even when the '
        'user has separate_folder_per_channel on'
    )


@pytest.mark.parametrize('run_mode', _AUTOFOCUS_RUN_MODES, ids=lambda m: m.value)
def test_autofocus_scan_plan_carries_the_forced_values(run_mode):
    """The forced values must survive the whole starter chain: helper ->
    prepare() -> RunPlan, which is what the run actually reads."""
    from tests.protocol_drives import bare_capture_runner, scr_run_kwargs

    run_settings = config_helpers.get_sequenced_run_settings(
        {
            'keep_led_between_steps': True,
            'video_as_frames': True,
            'separate_folder_per_channel': True,
        },
        run_mode=run_mode,
    )
    runner = bare_capture_runner()
    plan = runner.prepare(**{**scr_run_kwargs(run_mode=run_mode, max_scans=1), **run_settings})
    assert plan.keep_led_between_steps is False, (
        'the AF-scan plan must carry keep_led_between_steps=False'
    )
    assert plan.separate_folder_per_channel is False, (
        'the AF-scan plan must carry separate_folder_per_channel=False'
    )


def test_a_composite_captures_8_bit_in_a_format_its_merge_reads():
    """The composite's image rule is the helper's, like the autofocus
    overrides: its merge reads its inputs back as 8-bit, from a TIFF or an
    OME-TIFF, whatever the user's image mode and sequenced format."""
    settings = {
        'image_mode': IMAGE_MODE_12BIT_SCIENTIFIC,
        'image_output_format': {'live': 'TIFF', 'sequenced': OUTPUT_FORMAT_HYPERSTACK},
    }
    composite = config_helpers.get_sequenced_run_settings(
        settings, run_mode=SequencedCaptureRunMode.SINGLE_COMPOSITE
    )['image_capture_config']
    scan = config_helpers.get_sequenced_run_settings(
        settings, run_mode=SequencedCaptureRunMode.SINGLE_SCAN
    )['image_capture_config']
    assert composite.image_mode == IMAGE_MODE_8BIT
    assert composite.output_format_sequenced == OUTPUT_FORMAT_OME_TIFF
    assert scan.image_mode == IMAGE_MODE_12BIT_SCIENTIFIC
    assert scan.output_format_sequenced == OUTPUT_FORMAT_HYPERSTACK


def test_every_acquisition_run_call_is_classified():
    # The scanner finds the one known site, so the guards below are not
    # vacuously passing; and there is one: ProtocolRunner's prepare(), which
    # every member reaches and every GUI run button starts through. A second
    # site is a second way to start a run.
    sites = [rel for rel, _ in _acquisition_run_call_blocks()]
    assert sites == ['modules/protocol_runner.py'], (
        f'expected the one acquisition prepare() site in ProtocolRunner, found {sites}'
    )


def test_every_acquisition_site_spreads_the_helper():
    for rel, block in _acquisition_run_call_blocks():
        assert _SPREAD in block, (
            f'{rel}: an acquisition prepare() call does not spread '
            f'get_sequenced_run_settings(); a settings-derived run param can '
            f'drift on this path (this is how the GUI scan/z-stack bug shipped).'
        )


def test_every_helper_spread_passes_its_run_mode():
    # The helper resolves two of its values from the run mode, so a spread
    # that omits run_mode= is not merely a style slip: it cannot be written
    # (the parameter is required) and, if it ever became optional, an AF
    # scan would silently inherit the user's LED-hold setting.
    for rel, block in _acquisition_run_call_blocks():
        idx = block.find(_HELPER_CALL)
        assert idx != -1, f'{rel}: acquisition prepare() call does not spread the helper'
        call = _balanced_block(block, idx + len(_HELPER_CALL) - 1)
        assert 'run_mode=' in call, (
            f'{rel}: get_sequenced_run_settings() is spread without run_mode=; '
            f'the helper cannot apply the autofocus-scan LED/folder safety '
            f'without knowing the run mode.'
        )


def test_keep_led_between_steps_never_hand_passed_to_run():
    # The regressed param must arrive only via the helper spread, never as a
    # per-site kwarg on a prepare() call -- inline or multiline, AF path included.
    for rel, block in _acquisition_run_call_blocks():
        assert not re.search(r'keep_led_between_steps\s*=', block), (
            f'{rel}: a prepare() call hand-passes keep_led_between_steps= instead '
            f'of sourcing it from get_sequenced_run_settings(); re-divergence risk.'
        )
