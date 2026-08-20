"""Invariant net for the camera-write authority consolidation.

Each camera-state setter on ImagingAPI emits a precise, ordered sequence of
frame-validity operations (invalidate + set_target) plus a cache-snapshot
update. An upcoming refactor routes every setter through one ``_camera_write``
authority so a future setter cannot forget to invalidate. That refactor must be
behavior-preserving: the emitted sequence below is the contract it may not
change.

These tests run the real setters against a SimulatedCamera (the production
driver path) and record the exact validity-op sequence via a spy on the live
FrameValidity instance. They pass on the pre-refactor code and must stay green
through every migration commit. If a migration alters any sequence, the
matching test fails -- that is the regression catch.

The two SDK-perf setters (conversion gain mode, line noise reduction) are
Pylon-only; SimulatedCamera does not implement them, so a capable subclass adds
them here to pin their success path, and the plain sim pins the
driver-lacks-method path.
"""

from __future__ import annotations

import ast
import threading

import pytest

from drivers.simulated_camera import SimulatedCamera
from modules.exceptions import CameraSettingRejected
from modules.lumascope_api import Lumascope
from modules.lumascope_api.imaging import ImagingAPI
from tests.ast_seams import parse_module

# Driver methods that mutate camera state. A call to any of these must be wrapped
# in a write thunk handed to ImagingAPI._camera_write, never issued directly --
# that is what couples every camera write to its frame-validity invalidation.
CAMERA_WRITE_METHODS = frozenset(
    {
        'gain',
        'exposure_t',
        'auto_gain',
        'auto_exposure_t',
        'auto_gain_once',
        'set_frame_size',
        'set_binning_size',
        'set_pixel_format',
        'set_conversion_gain_mode',
        'set_line_noise_reduction',
        'update_auto_gain_target_brightness',
    }
)

_IMAGING_REL = 'modules/lumascope_api/imaging.py'


class _CamWriteCapableSim(SimulatedCamera):
    """SimulatedCamera plus the two Pylon-only SDK setters, so the success
    path of set_conversion_gain_mode / set_line_noise_reduction (driver
    implements the method and returns True) is exercisable in the sim."""

    def __init__(self):
        super().__init__()
        self._conversion_gain_mode = 'Low'
        self._line_noise_reduction = False

    def set_conversion_gain_mode(self, mode: str) -> bool:
        self._conversion_gain_mode = mode
        return True

    def set_line_noise_reduction(self, enabled: bool) -> bool:
        self._line_noise_reduction = enabled
        return True


def _build_imaging(cam):
    cam.active = True
    scope = Lumascope.__new__(Lumascope)
    scope._camera_driver = cam
    scope._camera_executor = None
    scope._cam_lock = threading.RLock()
    scope._state_lock = threading.RLock()
    imaging = ImagingAPI(scope, cam)
    scope.imaging = imaging
    return imaging


@pytest.fixture
def imaging_capable():
    """ImagingAPI on a sim that implements every camera setter."""
    return _build_imaging(_CamWriteCapableSim())


@pytest.fixture
def imaging_plain():
    """ImagingAPI on a stock SimulatedCamera (no conversion-gain / line-noise)."""
    return _build_imaging(SimulatedCamera())


def _record_validity_events(imaging):
    """Patch invalidate + set_target on imaging.frame_validity to append an
    ordered event log, then return the (still-live) log list. Each event is
    ('invalidate', source) or ('set_target', source, value); the real method
    still runs so downstream state stays correct.
    """
    events = []
    fv = imaging.frame_validity
    orig_invalidate = fv.invalidate
    orig_set_target = fv.set_target

    def recording_invalidate(source):
        events.append(('invalidate', source))
        return orig_invalidate(source)

    def recording_set_target(source, value):
        events.append(('set_target', source, value))
        return orig_set_target(source, value)

    fv.invalidate = recording_invalidate
    fv.set_target = recording_set_target
    return events


class TestValueSetterSequences:
    """Manual value setters: invalidate the source, then record the chunk
    target. Both fire on every successful write (never gated by the cache)."""

    def test_set_gain_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable.set_gain(7.0)
        assert events == [
            ('invalidate', 'gain'),
            ('set_target', 'gain', 7.0),
        ]
        assert imaging_capable.gain_cached == pytest.approx(7.0)

    def test_set_exposure_time_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable.set_exposure_ms(0.1)
        # Target is recorded in microseconds (chunk-match unit); API takes ms.
        assert events == [
            ('invalidate', 'exposure'),
            ('set_target', 'exposure', 100.0),
        ]
        assert imaging_capable.exposure_ms_cached == pytest.approx(0.1)


class TestAutoSetterSequences:
    """Auto/mode setters flip a mode node while the value node is unchanged.
    Their invalidations are unconditional (force) -- the auto_gain settle
    window arms only because invalidate('auto_gain') fires here."""

    def test_set_auto_gain_enable_arms_settle_window(self, imaging_capable):
        # The write authority emits all invalidations before the target-clear,
        # so 'auto_gain' invalidate precedes the gain target-clear here. That
        # reorder vs the pre-authority code is semantically inert: invalidate
        # and set_target write independent frame_validity entries (pending vs
        # target), and the only load-bearing order -- write before invalidate --
        # is preserved. The arm itself (invalidate('auto_gain')) is what matters.
        cam = imaging_capable._driver
        expected = [('invalidate', 'gain')]
        if getattr(cam.profile, 'has_auto_gain', False):
            expected.append(('invalidate', 'auto_gain'))
        expected.append(('set_target', 'gain', None))
        events = _record_validity_events(imaging_capable)
        imaging_capable.set_auto_gain(
            True,
            {
                'target_brightness': 0.5,
                'min_gain_db': 0.0,
                'max_gain_db': 24.0,
                'max_exposure_ms': 100.0,
            },
        )
        assert events == expected

    def test_set_auto_gain_disable_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable.set_auto_gain(
            False,
            {
                'target_brightness': 0.5,
                'min_gain_db': 0.0,
                'max_gain_db': 24.0,
                'max_exposure_ms': 100.0,
            },
        )
        # Disable does not arm the auto_gain window; it clears the gain target.
        assert events == [('invalidate', 'gain'), ('set_target', 'gain', None)]

    def test_update_auto_gain_target_brightness_arms_settle_window(self, imaging_capable):
        # A target change re-drives the running AG convergence, so it marks the
        # same settle sources set_auto_gain arms -- forced (never gated on a
        # value delta), and no manual target is recorded (gain stays AG-driven).
        cam = imaging_capable._driver
        expected = [('invalidate', 'gain')]
        if getattr(cam.profile, 'has_auto_gain', False):
            expected.append(('invalidate', 'auto_gain'))
        events = _record_validity_events(imaging_capable)
        imaging_capable.update_auto_gain_target_brightness(0.5)
        assert events == expected

    def test_set_auto_exposure_enable_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable.set_auto_exposure_time(True)
        assert events == [('invalidate', 'exposure'), ('set_target', 'exposure', None)]

    def test_set_auto_exposure_disable_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable.set_auto_exposure_time(False)
        assert events == [('invalidate', 'exposure'), ('set_target', 'exposure', None)]

    def test_auto_gain_once_invalidates_both_sources(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable.auto_gain_once(
            state=True,
            target_brightness=0.5,
            min_gain_db=0.0,
            max_gain_db=24.0,
            ae_max_exposure_ms=100.0,
        )
        assert events == [
            ('invalidate', 'gain'),
            ('invalidate', 'exposure'),
            ('set_target', 'gain', None),
            ('set_target', 'exposure', None),
        ]


class TestGeometrySetterSequences:
    """Geometry setters invalidate one source; pixel_format and frame_size
    also snapshot the cache."""

    def test_set_frame_size_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable.set_frame_size(640, 482)
        assert events == [('invalidate', 'frame_size')]
        # The cache holds the DELIVERED size, not the request: the sim snaps
        # 640 -> 624 (48 grid) and 482 -> 480 (4 grid).
        assert imaging_capable.frame_size_cached == {'width': 624, 'height': 480}

    def test_set_frame_size_caches_delivered_geometry(self, imaging_capable):
        # The sim snaps width to a 48 grid and height to a 4 grid, so the
        # cache must hold the snapped size the driver delivered (via the
        # write's own return value), not the request.
        imaging_capable.set_frame_size(640, 482)
        assert imaging_capable.frame_size_cached == {'width': 624, 'height': 480}

    def test_set_frame_size_rejected_write_keeps_prior_cache(self, imaging_capable):
        imaging_capable.set_frame_size(624, 480)
        events = _record_validity_events(imaging_capable)
        cam = imaging_capable._driver
        orig = cam.set_frame_size
        cam.set_frame_size = lambda w, h: False
        try:
            # Rejection surfaces as the typed raise (the apply contract);
            # the sequencing assertions below are the test's subject.
            with pytest.raises(CameraSettingRejected):
                imaging_capable.set_frame_size(1200, 800)
        finally:
            cam.set_frame_size = orig
        # A rejected write still expires validity (force-invalidate), but the
        # cache keeps the geometry the hardware still has.
        assert events == [('invalidate', 'frame_size')]
        assert imaging_capable.frame_size_cached == {'width': 624, 'height': 480}

    def test_set_binning_size_success_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        result = imaging_capable.set_binning_size(2)
        assert result is True
        assert events == [('invalidate', 'binning')]

    def test_binning_read_failure_sentinel_not_committed(self, imaging_capable):
        imaging_capable.set_binning_size(2)
        cam = imaging_capable._driver
        orig = cam.get_binning_size
        cam.get_binning_size = lambda: -1
        try:
            imaging_capable._populate_camera_cache()
        finally:
            cam.get_binning_size = orig
        # A failed read returns the out-of-band -1 sentinel and must leave the
        # last-known factor in place -- committing it (or an in-band 1) would
        # silently de-bin the scale-bar / FOV math for a 2x camera.
        assert imaging_capable._binning_size == 2

    def test_gain_exposure_read_failure_not_cached(self, imaging_capable):
        prior_gain = imaging_capable.gain_cached
        prior_exposure = imaging_capable.exposure_ms_cached
        cam = imaging_capable._driver
        orig_gain, orig_exp = cam.get_gain, cam.get_exposure_t
        cam.get_gain = lambda: -1.0
        cam.get_exposure_t = lambda: -1.0
        try:
            imaging_capable._populate_camera_cache()
        finally:
            cam.get_gain, cam.get_exposure_t = orig_gain, orig_exp
        # A negative return is the drivers' failed-read sentinel. The old
        # `or 0.0` idiom passed it through (-1.0 is truthy) and latched -1
        # into the UI gain/exposure readout; a failed read must leave the
        # last-known values in place.
        assert imaging_capable.gain_cached == prior_gain
        assert imaging_capable.exposure_ms_cached == prior_exposure
        assert imaging_capable.gain_cached >= 0
        assert imaging_capable.exposure_ms_cached >= 0

    def test_rejected_binning_write_keeps_prior_factor(self, imaging_capable):
        imaging_capable.set_binning_size(2)
        cam = imaging_capable._driver
        orig = cam.set_binning_size
        cam.set_binning_size = lambda size: False
        try:
            # Rejection surfaces as the typed raise (the apply contract).
            with pytest.raises(CameraSettingRejected):
                imaging_capable.set_binning_size(4)
        finally:
            cam.set_binning_size = orig
        # A rejected write must not commit the requested factor: the hardware
        # is still at the previous binning and scale-bar math reads this value.
        assert imaging_capable._binning_size == 2

    def test_set_binning_size_refreshes_geometry_caches(self, imaging_capable):
        imaging_capable.set_frame_size(1920, 1200)
        events = _record_validity_events(imaging_capable)
        result = imaging_capable.set_binning_size(2)
        assert result is True
        assert events == [('invalidate', 'binning')]
        # Binning 2x halves the sim's post-binning ceiling (1920x1200 native)
        # and the driver clamps the current frame down to it; both
        # binning-dependent geometry caches must reflect the driver's
        # post-binning reality, not the 1x values.
        assert imaging_capable.frame_size_cached == {'width': 960, 'height': 600}
        assert imaging_capable.min_frame_size_cached == (
            imaging_capable._driver.get_min_frame_size()
        )

    def test_set_pixel_format_success_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        result = imaging_capable.set_pixel_format('Mono8')
        assert result is True
        assert events == [('invalidate', 'pixel_format')]
        assert imaging_capable.pixel_format_cached == 'Mono8'


class TestSdkPerfSetterSequences:
    """Pylon-only setters: invalidate only when the driver implements the
    method AND returns truthy; a driver lacking the method returns False with
    no invalidation."""

    def test_set_conversion_gain_mode_success_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        result = imaging_capable.set_conversion_gain_mode('High')
        assert result is True
        assert events == [('invalidate', 'conversion_gain_mode')]

    def test_set_line_noise_reduction_success_sequence(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        result = imaging_capable.set_line_noise_reduction(True)
        assert result is True
        assert events == [('invalidate', 'line_noise_reduction')]

    def test_conversion_gain_mode_no_method_no_invalidate(self, imaging_plain):
        events = _record_validity_events(imaging_plain)
        result = imaging_plain.set_conversion_gain_mode('High')
        assert result is False
        assert events == []

    def test_line_noise_reduction_no_method_no_invalidate(self, imaging_plain):
        events = _record_validity_events(imaging_plain)
        result = imaging_plain.set_line_noise_reduction(True)
        assert result is False
        assert events == []


class TestCameraWriteAuthority:
    """The _camera_write authority in isolation: force vs applied-gated
    invalidation, target + cache maintenance, result-gating, and order."""

    def test_force_invalidate_fires_even_on_rejection(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        result = imaging_capable._camera_write(
            lambda: False,
            force_invalidate=('gain',),
            targets=(('gain', 5.0),),
        )
        # Rejection (False): force_invalidate still fires; the applied-only
        # target is suppressed.
        assert result is False
        assert events == [('invalidate', 'gain')]

    def test_applied_block_runs_when_not_rejected(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable._camera_write(
            lambda: None,
            force_invalidate=('gain',),
            targets=(('gain', 5.0),),
            cache_update={'gain_db': 5.0},
        )
        # None result counts as applied: force invalidate, then target + cache.
        assert events == [('invalidate', 'gain'), ('set_target', 'gain', 5.0)]
        assert imaging_capable.gain_cached == pytest.approx(5.0)

    def test_applied_gated_invalidate_skipped_on_false(self, imaging_capable):
        # The applied-only invalidate (not force_invalidate) is suppressed when
        # the driver returns False -- the result-gated setters' rejection path.
        events = _record_validity_events(imaging_capable)
        result = imaging_capable._camera_write(
            lambda: False,
            invalidates=('binning',),
        )
        assert result is False
        assert events == []

    def test_applied_gated_invalidate_fires_on_true(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        result = imaging_capable._camera_write(
            lambda: True,
            invalidates=('binning',),
        )
        assert result is True
        assert events == [('invalidate', 'binning')]

    def test_force_precedes_applied_invalidate_in_order(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable._camera_write(
            lambda: None,
            force_invalidate=('gain',),
            invalidates=('auto_gain',),
        )
        assert events == [('invalidate', 'gain'), ('invalidate', 'auto_gain')]

    def test_multiple_sources_and_targets(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        imaging_capable._camera_write(
            lambda: None,
            force_invalidate=('gain', 'exposure'),
            targets=(('gain', None), ('exposure', None)),
        )
        assert events == [
            ('invalidate', 'gain'),
            ('invalidate', 'exposure'),
            ('set_target', 'gain', None),
            ('set_target', 'exposure', None),
        ]

    def test_force_clear_fires_even_on_rejection(self, imaging_capable):
        events = _record_validity_events(imaging_capable)
        result = imaging_capable._camera_write(
            lambda: False,
            force_invalidate=('exposure',),
            force_clear=('exposure',),
            targets=(('gain', 5.0),),
        )
        # On rejection: force_invalidate and force_clear still fire (the
        # cleared target is None); the applied-gated target is suppressed.
        assert result is False
        assert events == [('invalidate', 'exposure'), ('set_target', 'exposure', None)]


def _imaging_tree():
    # The suite-wide cached parser, so future hardening of production-source AST
    # reads (encoding, AsyncFunctionDef, path resolution) does not bypass this
    # guard. The negative tests below still ast.parse() inline source strings.
    return parse_module(_IMAGING_REL)


def _parent_map(tree):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _find_funcdef(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _is_attr_call(node, owner_attr, method_set):
    """Match self.<owner_attr>.<method>(...) where method in method_set."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr not in method_set:
        return None
    owner = node.func.value
    if (
        isinstance(owner, ast.Attribute)
        and owner.attr == owner_attr
        and isinstance(owner.value, ast.Name)
        and owner.value.id == 'self'
    ):
        return node.func.attr
    return None


def _camera_write_calls(tree):
    """Every self._camera_write(...) call node."""
    calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == '_camera_write'
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'self'
        ):
            calls.append(node)
    return calls


def _enclosing_thunk(node, parents):
    """Nearest enclosing Lambda or FunctionDef above node."""
    cur = parents.get(id(node))
    while cur is not None:
        if isinstance(cur, (ast.Lambda, ast.FunctionDef)):
            return cur
        cur = parents.get(id(cur))
    return None


def _invalidate_offenders(tree):
    """Line numbers where self.frame_validity.invalidate(...) is called outside
    the _camera_write authority method."""
    authority = _find_funcdef(tree, '_camera_write')
    inside = set(map(id, ast.walk(authority))) if authority is not None else set()
    return [
        node.lineno
        for node in ast.walk(tree)
        if _is_attr_call(node, 'frame_validity', {'invalidate'}) and id(node) not in inside
    ]


def _driver_write_offenders(tree):
    """(offenders, found_count): camera driver writes not wrapped in a thunk
    handed to self._camera_write."""
    parents = _parent_map(tree)
    wired_lambdas = set()
    wired_closure_names = set()
    for call in _camera_write_calls(tree):
        for arg in call.args:
            if isinstance(arg, ast.Lambda):
                wired_lambdas.add(id(arg))
            elif isinstance(arg, ast.Name):
                wired_closure_names.add(arg.id)

    offenders = []
    found = 0
    for node in ast.walk(tree):
        method = _is_attr_call(node, '_driver', CAMERA_WRITE_METHODS)
        if method is None:
            continue
        found += 1
        thunk = _enclosing_thunk(node, parents)
        wired = (isinstance(thunk, ast.Lambda) and id(thunk) in wired_lambdas) or (
            isinstance(thunk, ast.FunctionDef) and thunk.name in wired_closure_names
        )
        if not wired:
            offenders.append((node.lineno, method))
    return offenders, found


class TestAuthorityIsSingleWritePath:
    """The structural guard that makes the omission unrepresentable: a new
    camera setter physically cannot write a camera node or invalidate a camera
    source outside the _camera_write authority. These AST checks fail the build
    if a future edit reintroduces a raw write or a raw invalidate."""

    def test_invalidate_only_inside_camera_write(self):
        offenders = _invalidate_offenders(_imaging_tree())
        assert not offenders, (
            f'self.frame_validity.invalidate(...) called outside _camera_write at '
            f'lines {offenders}; all camera invalidation must route through the authority '
            f'so a write and its invalidation are declared together.'
        )

    def test_camera_driver_writes_wrapped_in_authority_thunk(self):
        offenders, found = _driver_write_offenders(_imaging_tree())
        assert not offenders, (
            f'camera driver write issued outside a _camera_write thunk: {offenders}; '
            f'wrap the write in a lambda/closure handed to self._camera_write so it '
            f'declares its invalidation.'
        )
        # Guard against the check silently passing on zero findings (e.g. a
        # method-set drift): every migrated setter must still be seen.
        assert found >= len(CAMERA_WRITE_METHODS), (
            f'expected at least {len(CAMERA_WRITE_METHODS)} camera driver writes, found '
            f'{found}; the CAMERA_WRITE_METHODS set may be stale.'
        )

    def test_guard_detects_a_raw_invalidate(self):
        # The guard must BITE: a setter that invalidates a camera source directly
        # (outside the authority) is flagged.
        src = (
            'class ImagingAPI:\n'
            '    def _camera_write(self, write_fn):\n'
            '        return write_fn()\n'
            '    def set_thing(self):\n'
            "        self.frame_validity.invalidate('gain')\n"
        )
        assert _invalidate_offenders(ast.parse(src))

    def test_guard_detects_a_raw_driver_write(self):
        # A driver write issued directly, not wrapped in a _camera_write thunk.
        src = (
            'class ImagingAPI:\n'
            '    def _camera_write(self, write_fn):\n'
            '        return write_fn()\n'
            '    def set_thing(self):\n'
            '        self._driver.gain(0)\n'
        )
        offenders, found = _driver_write_offenders(ast.parse(src))
        assert offenders and found == 1

    def test_guard_accepts_a_wrapped_driver_write(self):
        # A driver write wrapped in a lambda handed to _camera_write is clean.
        src = (
            'class ImagingAPI:\n'
            '    def _camera_write(self, write_fn, **kw):\n'
            '        return write_fn()\n'
            '    def set_thing(self):\n'
            "        self._camera_write(lambda: self._driver.gain(0), force_invalidate=('gain',))\n"
        )
        offenders, found = _driver_write_offenders(ast.parse(src))
        assert not offenders and found == 1
