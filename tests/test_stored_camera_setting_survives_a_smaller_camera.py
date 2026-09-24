"""A stored gain/exposure the attached camera cannot reach is kept, not shrunk.

A per-channel gain or exposure in the settings store is the user's committed
intent and outlives whichever camera is attached. Connecting a smaller body
used to rewrite the store down to that body's cap, and the periodic
current.json flush persisted the loss with no record of the original -- so a
user who moved a scope between cameras silently lost what they had set.

The cap now applies to the value written to HARDWARE, decided in one place at
the API. The channel still runs at the most the body can do (the blackout that
motivated the original clamp is prevented on the apply, where it belongs), and
putting a capable camera back applies the stored intent again.

The API half is driven against a real SimulatedCamera rather than a stub, so
the assertions are about what actually reaches a driver. The ui/ half touches
Kivy widgets and cannot be imported under the test mocks, so those contracts
are pinned with AST guards, matching the other ui/ guards in this suite.
"""

import ast
import pathlib
import threading

import pytest

from drivers.simulated_camera import SimulatedCamera
from modules.lumascope_api._lumascope import Lumascope
from modules.lumascope_api.imaging import AppliedCameraSetting, ImagingAPI, cap_stored_value

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
IMAGE_SETTINGS_PATH = REPO_ROOT / 'ui' / 'image_settings.py'
LAYER_CONTROL_PATH = REPO_ROOT / 'ui' / 'layer_control.py'


def _func(path: pathlib.Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f'{name} not found in {path}')


@pytest.fixture
def imaging():
    """ImagingAPI on a real simulated camera, with known caps."""
    cam = SimulatedCamera()
    cam.active = True
    scope = Lumascope.__new__(Lumascope)
    scope._camera_driver = cam
    scope._camera_executor = None
    scope._cam_lock = threading.RLock()
    scope._state_lock = threading.RLock()
    api = ImagingAPI(scope, cam)
    scope.imaging = api
    with api._camera_cache_lock:
        api._camera_cache['max_gain_db'] = 20.0
        api._camera_cache['max_exposure_ms'] = 200.0
    return api


class TestTheApiDecidesWhatAStoredValueApplies:
    def test_a_value_over_the_cap_applies_as_the_cap_and_keeps_the_intent(self, imaging):
        gain = imaging.applied_gain_db_for(48.0)
        assert (gain.stored, gain.applied, gain.capped) == (48.0, 20.0, True)

    def test_a_value_under_the_cap_is_untouched(self, imaging):
        gain = imaging.applied_gain_db_for(10.0)
        assert (gain.stored, gain.applied, gain.capped) == (10.0, 10.0, False)

    def test_exposure_carries_the_same_contract(self, imaging):
        exposure = imaging.applied_exposure_ms_for(500.0)
        assert (exposure.stored, exposure.applied, exposure.capped) == (500.0, 200.0, True)

    def test_an_unknown_cap_narrows_nothing(self):
        # No camera, or a driver that publishes no maximum: a missing bound is
        # not a bound of zero, and inventing one would apply a limit no
        # hardware asked for.
        assert cap_stored_value(5.0, None) == AppliedCameraSetting(
            stored=5.0, applied=5.0, capped=False
        )

    def test_a_triple_that_contradicts_itself_cannot_be_built(self):
        # capped is whether the write differs from the intent, not a caller's
        # opinion: a consumer rendering one while testing the other would
        # report a limit that is not being applied.
        with pytest.raises(ValueError):
            AppliedCameraSetting(stored=48.0, applied=48.0, capped=True)
        with pytest.raises(ValueError):
            AppliedCameraSetting(stored=48.0, applied=20.0, capped=False)


class TestTheCappedValueIsWhatReachesTheDriver:
    def test_the_layer_apply_sends_the_cap_not_the_stored_value(self, imaging):
        # Driven against the real driver: a stub obeying the contract would
        # pass while the production path sent 48 dB and the camera refused
        # (pylon) or silently self-clamped while reporting success (IDS, FX2).
        seen = []
        original_gain, original_exposure = imaging._driver.gain, imaging._driver.exposure_t
        imaging._driver.gain = lambda v: (seen.append(('gain', v)), original_gain(v))[1]
        imaging._driver.exposure_t = lambda v: (seen.append(('exp', v)), original_exposure(v))[1]

        imaging._apply_layer_camera_settings_impl(48.0, 500.0, layer='Blue')

        assert seen == [('gain', 20.0), ('exp', 200.0)]


class TestTheGuiNeverNarrowsTheStore:
    def test_the_reconcile_writes_neither_stored_value(self):
        fn = _func(IMAGE_SETTINGS_PATH, 'reconcile_layers_to_camera_caps')
        written = {
            t.slice.value
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Subscript)
            and isinstance(t.slice, ast.Constant)
            and t.slice.value in ('gain_db', 'exposure_ms')
        }
        assert written == set(), f'the reconcile must not write the store; found {sorted(written)}'

    def test_an_unedited_text_box_commits_nothing(self):
        # The kv fires the text handlers on focus LOSS, not on edit, so a
        # click into a box and out again arrives with the untouched stored
        # value. Clipping that to the widget's bound is how a preserved
        # setting got destroyed by a stray click.
        fn = _func(LAYER_CONTROL_PATH, '_validate_and_apply_text_input')
        clip_lines = [
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'clip'
        ]
        assert clip_lines, 'expected the typed-value clip to still exist for edited input'
        guard_lines = [
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Compare)
            and any(isinstance(c, ast.Eq) for c in n.test.ops)
            and any(isinstance(b, ast.Return) for b in n.body)
        ]
        assert guard_lines, (
            '_validate_and_apply_text_input must return without committing when the '
            'parsed value already equals the stored one.'
        )
        assert min(guard_lines) < min(clip_lines), (
            f'the no-edit return (line {min(guard_lines)}) must precede the clip '
            f'(line {min(clip_lines)}), or an unedited box is still narrowed.'
        )

    def test_the_auto_gain_write_back_stores_the_apis_own_answer(self):
        # ImagingAPI.stored_exposure_after_lock decides this value so that a
        # GUI and a REST caller store the same thing; narrowing it again to a
        # slider's range makes the widget a second answerer over the API.
        fn = _func(LAYER_CONTROL_PATH, 'update_auto_gain_cb')
        clips = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'clip'
        ]
        assert not clips, 'the auto-gain write-back must not re-clip the API-decided value'
        assert any(
            isinstance(n, ast.Attribute) and n.attr == 'stored_exposure_ms' for n in ast.walk(fn)
        ), 'the auto-gain write-back must store the API-decided stored_exposure_ms'
