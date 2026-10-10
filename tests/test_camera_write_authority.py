"""Every camera-state write declares its capture-readiness consequence.

``ImagingAPI._camera_write`` is the one path a camera setter writes through,
so a write and the frame-validity invalidation it requires are declared
together and a new setter cannot write a camera node and forget to
invalidate. Frame validity (``modules/frame_validity.py``) is the owner of
capture readiness: tracked state only, the changer invalidates its source and
the capture path drains until the count is met.

The behavioural tests drive the public setters on a simulated scope and read
the owner through its public reads: ``invalidation_counts`` (which sources a
setter invalidated, and that it invalidated no other), ``pending_sources``
and ``target(source)`` (the value a frame's chunk is checked against). Nothing
grabs a frame in these tests, so nothing the setters left pending can clear
before it is read. The AST guards below make the omission unrepresentable.

The two SDK-perf setters (conversion gain mode, line noise reduction) are
Pylon-only; SimulatedCamera does not implement them, so a capable subclass adds
them here to pin their success path, and the plain simulator pins the
camera-lacks-the-mode path.
"""

from __future__ import annotations

import ast

import pytest

from drivers.simulated_camera import SimulatedCamera
from modules.exceptions import (
    CameraSettingOutOfRangeError,
    CameraSettingRejected,
    CameraSettingUnsupportedError,
    HardwareCommandRefusedError,
)
from modules.lumascope_api import Lumascope
from modules.lumascope_api import _lumascope
from tests.ast_seams import parse_module
from tests.scope_fakes import bind_settings_like_a_session, build_scope

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
        'set_black_level',
        'update_auto_gain_target_brightness',
    }
)

_IMAGING_REL = 'modules/lumascope_api/imaging.py'

_AUTO_GAIN_SETTINGS = {
    'target_brightness': 0.5,
    'min_gain_db': 0.0,
    'max_gain_db': 24.0,
    'max_exposure_ms': 100.0,
}


class _CamWriteCapableSim(SimulatedCamera):
    """SimulatedCamera plus the two Pylon-only SDK setters and the probes
    that declare them, so the success path of set_conversion_gain_mode /
    set_line_noise_reduction is exercisable in the sim."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._conversion_gain_mode = 'Low'
        self._line_noise_reduction = False

    def set_conversion_gain_mode(self, mode: str) -> bool:
        self._conversion_gain_mode = mode
        return True

    def set_line_noise_reduction(self, enabled: bool) -> bool:
        self._line_noise_reduction = enabled
        return True

    def supports_conversion_gain_mode(self) -> bool:
        return True

    def supports_line_noise_reduction(self) -> bool:
        return True


def _sim_imaging():
    """The imaging API of a simulated scope, bound to the shipped settings."""
    scope = build_scope(simulate=True)
    bind_settings_like_a_session(scope)
    return scope.imaging


@pytest.fixture
def imaging_plain():
    """A simulated scope's imaging, on its stock SimulatedCamera."""
    return _sim_imaging()


@pytest.fixture
def imaging_capable(monkeypatch):
    """A simulated scope's imaging, on a camera with every camera setter."""

    def _capable_camera(name='auto', *, simulate=False, **kwargs):
        return _CamWriteCapableSim(**kwargs)

    # a stand-in by design: SimulatedCamera has no conversion-gain or line-noise-reduction node (Pylon-only); the simulator gap stage 7 fills, and this subclass then goes
    monkeypatch.setattr(_lumascope.camera_registry, 'create', _capable_camera)
    return _sim_imaging()


def _grown_by_one(before, *sources):
    """``before`` with each of ``sources`` invalidated once more: what
    ``invalidation_counts`` reads after a write that invalidated exactly
    those sources."""
    after = dict(before)
    for source in sources:
        after[source] = after.get(source, 0) + 1
    return after


class TestValueSetterSequences:
    """Manual value setters record the value the camera took as the chunk
    target: the value ``chunk_match`` rejects a stale frame against."""

    def test_the_exposure_target_is_the_microseconds_the_camera_took(self, imaging_plain):
        """The setter takes milliseconds; the frame's ExposureTime chunk is in
        microseconds, so the target is recorded in microseconds. The target
        never clears a source: settling is by frame count alone."""
        imaging_plain.set_exposure_ms(2.5)
        assert imaging_plain.frame_validity.target('exposure') == 2500.0


class TestAutoSetterSequences:
    """Auto/mode setters flip a mode node while the value node is unchanged,
    so their invalidations are unconditional; with the camera driving the
    value, the manual chunk target is cleared and settling falls back to the
    frame count alone."""

    def test_set_auto_gain_enable_arms_settle_window(self, imaging_plain):
        """Arming continuous auto-gain holds readiness for 'gain' and for the
        'auto_gain' settle window, and clears the manual gain target."""
        fv = imaging_plain.frame_validity
        imaging_plain.set_gain_db(5.0)
        assert fv.target('gain') == 5.0, 'precondition: a manual gain target'
        before = fv.invalidation_counts
        imaging_plain.set_auto_gain(True, _AUTO_GAIN_SETTINGS)
        assert fv.invalidation_counts == _grown_by_one(before, 'gain', 'auto_gain')
        assert fv.target('gain') is None

    def test_set_auto_gain_disable_sequence(self, imaging_plain):
        """Disarming invalidates 'gain' and arms no 'auto_gain' window; the
        gain target stays cleared until a manual gain is written."""
        fv = imaging_plain.frame_validity
        imaging_plain.set_gain_db(5.0)
        before = fv.invalidation_counts
        imaging_plain.set_auto_gain(False, _AUTO_GAIN_SETTINGS)
        assert fv.invalidation_counts == _grown_by_one(before, 'gain')
        assert fv.target('gain') is None

    def test_update_auto_gain_target_brightness_arms_settle_window(self, imaging_plain):
        """A new target brightness re-drives the running convergence, so it
        holds readiness for the same sources arming does."""
        fv = imaging_plain.frame_validity
        before = fv.invalidation_counts
        imaging_plain.update_auto_gain_target_brightness(0.5)
        assert fv.invalidation_counts == _grown_by_one(before, 'gain', 'auto_gain')

    def test_set_auto_exposure_enable_sequence(self, imaging_plain):
        """Toggling auto-exposure holds readiness for 'exposure' and clears
        the manual exposure target."""
        fv = imaging_plain.frame_validity
        imaging_plain.set_exposure_ms(2.5)
        assert fv.target('exposure') == 2500.0, 'precondition: a manual exposure target'
        before = fv.invalidation_counts
        imaging_plain.set_auto_exposure_time(True)
        assert fv.invalidation_counts == _grown_by_one(before, 'exposure')
        assert fv.target('exposure') is None


class TestGeometrySetterSequences:
    """Geometry setters: the cache holds the geometry the camera delivered,
    and a rejected frame-size write still holds readiness."""

    def test_set_frame_size_caches_delivered_geometry(self, imaging_plain):
        # The cache must hold the size the driver delivered (via the write's
        # own return value), not the request: a driver answering a smaller
        # size (a request within one grid step of its maximum) is believed.
        cam = imaging_plain._driver
        cam.set_frame_size = lambda w, h: {'width': 624, 'height': 480}
        imaging_plain.set_frame_size(640, 482)
        assert imaging_plain.frame_size_cached == {'width': 624, 'height': 480}

    def test_set_frame_size_rejected_write_keeps_prior_cache(self, imaging_plain, monkeypatch):
        """A rejected frame-size write still invalidates 'frame_size' (the
        camera may have restarted its grab engine), and the cache keeps the
        geometry the camera still has."""
        imaging_plain.set_frame_size(624, 480)
        fv = imaging_plain.frame_validity
        before = fv.invalidation_counts
        # a stand-in by design: the simulated camera takes every legal frame size; a refused one is the gap stage 7 fills
        monkeypatch.setattr(imaging_plain._driver, 'set_frame_size', lambda w, h: False)
        with pytest.raises(CameraSettingRejected) as caught:
            imaging_plain.set_frame_size(1200, 800)
        assert caught.value.setting == 'frame_size'
        assert fv.invalidation_counts == _grown_by_one(before, 'frame_size')
        assert 'frame_size' in fv.pending_sources
        assert imaging_plain.frame_size_cached == {'width': 624, 'height': 480}

    def test_binning_read_failure_sentinel_not_committed(self, imaging_plain):
        imaging_plain.set_binning_size(2)
        cam = imaging_plain._driver
        orig = cam.get_binning_size
        cam.get_binning_size = lambda: -1
        try:
            imaging_plain._populate_camera_cache()
        finally:
            cam.get_binning_size = orig
        # A failed read returns the out-of-band -1 sentinel and must leave the
        # last-known factor in place -- committing it (or an in-band 1) would
        # silently de-bin the scale-bar / FOV math for a 2x camera.
        assert imaging_plain._binning_size == 2

    def test_gain_exposure_read_failure_not_cached(self, imaging_plain):
        prior_gain = imaging_plain.gain_db_cached
        prior_exposure = imaging_plain.exposure_ms_cached
        cam = imaging_plain._driver
        orig_gain, orig_exp = cam.get_gain, cam.get_exposure_t
        cam.get_gain = lambda: -1.0
        cam.get_exposure_t = lambda: -1.0
        try:
            imaging_plain._populate_camera_cache()
        finally:
            cam.get_gain, cam.get_exposure_t = orig_gain, orig_exp
        # A negative return is the drivers' failed-read sentinel. The old
        # `or 0.0` idiom passed it through (-1.0 is truthy) and latched -1
        # into the UI gain/exposure readout; a failed read must leave the
        # last-known values in place.
        assert imaging_plain.gain_db_cached == prior_gain
        assert imaging_plain.exposure_ms_cached == prior_exposure
        assert imaging_plain.gain_db_cached >= 0
        assert imaging_plain.exposure_ms_cached >= 0

    def test_rejected_binning_write_keeps_prior_factor(self, imaging_plain):
        imaging_plain.set_binning_size(2)
        cam = imaging_plain._driver
        orig = cam.set_binning_size
        cam.set_binning_size = lambda size: False
        try:
            # Rejection surfaces as the typed raise (the apply contract).
            with pytest.raises(CameraSettingRejected):
                imaging_plain.set_binning_size(4)
        finally:
            cam.set_binning_size = orig
        # A rejected write must not commit the requested factor: the hardware
        # is still at the previous binning and scale-bar math reads this value.
        assert imaging_plain._binning_size == 2

    def test_set_binning_size_refreshes_geometry_caches(self, imaging_plain):
        imaging_plain.set_frame_size(3840, 2160)
        imaging_plain.set_binning_size(2)
        # Binning 2x halves the sim's post-binning ceiling (3840x2160 native)
        # and the driver clamps the current frame down to it; both
        # binning-dependent geometry caches must reflect the driver's
        # post-binning reality, not the 1x values.
        assert imaging_plain.frame_size_cached == {'width': 1920, 'height': 1080}
        assert imaging_plain.min_frame_size_cached == (imaging_plain._driver.get_min_frame_size())


class TestSdkPerfSetterSequences:
    """The two camera toggles: a change the camera took holds readiness for
    its source; a refusal, of either kind, invalidates nothing."""

    def test_set_conversion_gain_mode_success_sequence(self, imaging_capable):
        imaging_capable.set_conversion_gain_mode('High')
        assert 'conversion_gain_mode' in imaging_capable.frame_validity.pending_sources
        assert imaging_capable._driver._conversion_gain_mode == 'High'

    def test_set_line_noise_reduction_success_sequence(self, imaging_capable):
        imaging_capable.set_line_noise_reduction(True)
        assert 'line_noise_reduction' in imaging_capable.frame_validity.pending_sources
        assert imaging_capable._driver._line_noise_reduction is True

    def test_conversion_gain_on_without_the_mode_is_refused(self, imaging_plain):
        """Refused before the camera: nothing changed, so readiness is untouched."""
        before = imaging_plain.frame_validity.invalidation_counts
        with pytest.raises(CameraSettingUnsupportedError) as caught:
            imaging_plain.set_conversion_gain_mode('High')
        assert caught.value.reason == 'conversion_gain_mode_unsupported'
        assert caught.value.offered == ('Low',)
        assert str(caught.value).startswith('This camera has no high conversion gain.')
        assert imaging_plain.frame_validity.invalidation_counts == before

    def test_line_noise_on_without_the_filter_is_refused(self, imaging_plain):
        """Refused before the camera: nothing changed, so readiness is untouched."""
        before = imaging_plain.frame_validity.invalidation_counts
        with pytest.raises(CameraSettingUnsupportedError) as caught:
            imaging_plain.set_line_noise_reduction(True)
        assert caught.value.reason == 'line_noise_reduction_unsupported'
        assert caught.value.offered == (False,)
        assert imaging_plain.frame_validity.invalidation_counts == before

    def test_off_without_the_mode_is_that_cameras_state(self, imaging_plain):
        """Off on a camera without the feature is its state, not a write, so
        readiness is untouched."""
        before = imaging_plain.frame_validity.invalidation_counts
        imaging_plain.set_conversion_gain_mode('Low')
        imaging_plain.set_line_noise_reduction(False)
        assert imaging_plain.frame_validity.invalidation_counts == before

    def test_a_mode_that_is_neither_high_nor_low_is_refused(self, imaging_capable):
        """Refused before the camera: nothing changed, so readiness is untouched."""
        before = imaging_capable.frame_validity.invalidation_counts
        with pytest.raises(CameraSettingUnsupportedError) as caught:
            imaging_capable.set_conversion_gain_mode('Bogus')
        assert caught.value.offered == ('High', 'Low')
        assert imaging_capable._driver._conversion_gain_mode == 'Low'
        assert imaging_capable.frame_validity.invalidation_counts == before

    @pytest.mark.parametrize(
        ('member', 'driver_member', 'value'),
        [
            ('set_conversion_gain_mode', 'set_conversion_gain_mode', 'High'),
            ('set_line_noise_reduction', 'set_line_noise_reduction', True),
        ],
    )
    def test_a_driver_refusal_is_a_rejection(
        self, imaging_capable, monkeypatch, member, driver_member, value
    ):
        """A toggle the camera refused invalidates nothing. Pins today's
        applied-only invalidation; flips with ruling K12's build."""
        before = imaging_capable.frame_validity.invalidation_counts
        # a stand-in by design: the simulator cannot yet refuse a toggle; the fault stage 7 injects
        monkeypatch.setattr(imaging_capable._driver, driver_member, lambda *a, **k: False)
        with pytest.raises(CameraSettingRejected) as caught:
            getattr(imaging_capable, member)(value)
        assert caught.value.__cause__ is None
        assert imaging_capable.frame_validity.invalidation_counts == before

    @pytest.mark.parametrize(
        ('member', 'driver_member', 'value'),
        [
            ('set_conversion_gain_mode', 'set_conversion_gain_mode', 'High'),
            ('set_line_noise_reduction', 'set_line_noise_reduction', True),
        ],
    )
    def test_a_driver_raise_is_a_chained_rejection(
        self, imaging_capable, monkeypatch, member, driver_member, value
    ):
        """A toggle whose write raised invalidates nothing. Pins today's
        applied-only invalidation; flips with ruling K12's build."""
        before = imaging_capable.frame_validity.invalidation_counts
        fault = RuntimeError('node write failed')

        def raising(*a, **k):
            raise fault

        # a stand-in by design: the simulator cannot yet fail a node write; the fault stage 7 injects
        monkeypatch.setattr(imaging_capable._driver, driver_member, raising)
        with pytest.raises(CameraSettingRejected) as caught:
            getattr(imaging_capable, member)(value)
        assert caught.value.__cause__ is fault
        assert imaging_capable.frame_validity.invalidation_counts == before


class TestTheImplsAbsentAnswerIsRefused:
    """A camera gone between the lane's question and the body: the body's
    absent answer is the refusal naming the camera, never a bare False."""

    @pytest.mark.parametrize(
        ('member', 'value'),
        [
            ('set_binning_size', 2),
            ('set_pixel_format', 'Mono8'),
            ('set_conversion_gain_mode', 'High'),
            ('set_line_noise_reduction', True),
        ],
    )
    def test_absent_answer_raises_not_connected(self, imaging_plain, monkeypatch, member, value):
        # The lane's question answers connected; the driver drops before the body.
        monkeypatch.setattr(Lumascope, 'camera_connected', property(lambda self: True))
        imaging_plain._driver.active = False
        with pytest.raises(HardwareCommandRefusedError) as caught:
            getattr(imaging_plain, member)(value)
        assert caught.value.reason == 'not_connected'
        assert caught.value.member == member


class TestBlackLevelSetterSequence:
    """The black level holds readiness for its own source when the camera
    took the value, and invalidates nothing when it was refused before
    reaching the camera."""

    def test_set_black_level_success_sequence(self, imaging_plain):
        assert imaging_plain.set_black_level(4.0) == 4.0
        assert 'black_level' in imaging_plain.frame_validity.pending_sources

    def test_an_out_of_range_black_level_invalidates_nothing(self, imaging_plain):
        before = imaging_plain.frame_validity.invalidation_counts
        with pytest.raises(CameraSettingOutOfRangeError) as caught:
            imaging_plain.set_black_level(1000.0)
        assert caught.value.reason == 'black_level_out_of_range'
        assert imaging_plain.frame_validity.invalidation_counts == before


class TestCameraWriteAuthority:
    """The authority's two gates, through the value and mode setters: a
    forced source is invalidated and a forced target clear happens whether or
    not the camera took the write; a target is recorded only when it did, and
    a driver that cannot confirm (``None``) counts as having taken it."""

    def test_force_invalidate_fires_even_on_rejection(self, imaging_plain, monkeypatch):
        """A refused gain still holds readiness for 'gain', and records no
        target: the prior one stands."""
        fv = imaging_plain.frame_validity
        imaging_plain.set_gain_db(3.0)
        before = fv.invalidation_counts
        # a stand-in by design: the simulated gain node takes every in-range value; a refusal is the fault stage 7 injects
        monkeypatch.setattr(imaging_plain._driver, 'gain', lambda v: False)
        with pytest.raises(CameraSettingRejected) as caught:
            imaging_plain.set_gain_db(9.0)
        assert caught.value.setting == 'gain_db'
        assert fv.invalidation_counts == _grown_by_one(before, 'gain')
        assert fv.target('gain') == 3.0

    def test_applied_block_runs_when_not_rejected(self, imaging_plain, monkeypatch):
        """A driver that cannot confirm answers None, which is not a refusal:
        the request is recorded as the chunk target."""
        # a stand-in by design: a driver with no confirmation signal (answers None); the simulator always confirms
        monkeypatch.setattr(imaging_plain._driver, 'gain', lambda v: None)
        imaging_plain.set_gain_db(5.0)
        assert imaging_plain.frame_validity.target('gain') == 5.0

    def test_force_clear_fires_even_on_rejection(self, imaging_plain, monkeypatch):
        """A refused auto-exposure toggle still holds readiness for
        'exposure' and still clears the manual exposure target."""
        fv = imaging_plain.frame_validity
        imaging_plain.set_exposure_ms(2.5)
        before = fv.invalidation_counts
        # a stand-in by design: the simulated auto-exposure node always takes the toggle; a refusal is the fault stage 7 injects
        monkeypatch.setattr(imaging_plain._driver, 'auto_exposure_t', lambda state=True: False)
        with pytest.raises(CameraSettingRejected) as caught:
            imaging_plain.set_auto_exposure_time(True)
        assert caught.value.setting == 'auto_exposure'
        assert fv.invalidation_counts == _grown_by_one(before, 'exposure')
        assert fv.target('exposure') is None


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


# The camera drivers that implement the members above. A member here answers
# _camera_write, whose rule is that anything not False was applied -- so a
# member that swallows a rejection and falls off the end reports the refusal as
# a success, and the requested value is cached and stamped as the chunk target.
_CAMERA_DRIVER_RELS = (
    'drivers/camera.py',
    'drivers/pyloncamera.py',
    'drivers/simulated_camera.py',
    'drivers/fx2driver.py',
    'drivers/idscamera.py',
)


def _handler_reraises(handler):
    return any(isinstance(node, ast.Raise) for node in ast.walk(handler))


def _handler_reports_refused(handler):
    """Some path out of the handler returns an explicit False."""
    return any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and node.value.value is False
        for node in ast.walk(handler)
    )


def _swallowed_rejection_offenders(tree):
    """(offenders, found): camera-write members whose FINAL statement is a try
    with a handler that neither re-raises nor returns False.

    Scoped to the try that ENDS the member on purpose. A handler with code
    after it is tolerating one step and continuing -- pyloncamera.gain's
    GainSelector write is the sanctioned case, since cameras without the
    selector are expected. A handler that ends the member decides the member's
    answer, and falling off the end of one answers None, which the authority
    reads as applied. Annotations are deliberately NOT consulted: a member can
    be annotated -> bool and never return False, so the type declares an
    intention while the body decides the behaviour.
    """
    offenders = []
    found = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in CAMERA_WRITE_METHODS:
            continue
        found += 1
        last = node.body[-1] if node.body else None
        if not isinstance(last, ast.Try):
            continue
        for handler in last.handlers:
            if _handler_reraises(handler) or _handler_reports_refused(handler):
                continue
            offenders.append((node.name, handler.lineno))
    return offenders, found


class TestNoSetterSwallowsItsRejection:
    """A camera-write member may raise or report False, never both-nor.

    pyloncamera.gain caught four GenICam rejection classes -- all siblings of
    RuntimeException rather than subclasses, so they arrive with the camera
    still live -- logged them, and fell off the end returning None. The
    authority read that as applied and recorded a gain the hardware refused.
    """

    def test_no_camera_driver_swallows_a_rejection(self):
        all_offenders = []
        total_found = 0
        for rel in _CAMERA_DRIVER_RELS:
            offenders, found = _swallowed_rejection_offenders(parse_module(rel))
            total_found += found
            all_offenders += [(rel, name, line) for name, line in offenders]

        assert not all_offenders, (
            f'camera setter swallows an exception and answers None: {all_offenders}. '
            f'The handler that ENDS the member must return False (refused) or re-raise; '
            f'falling off the end reports the refusal to _camera_write as applied, which '
            f'caches the value and stamps it as the chunk target.'
        )
        # Never pass on zero findings: a rename or a member-set drift would
        # otherwise make this guard silently vacuous.
        assert total_found >= len(CAMERA_WRITE_METHODS), (
            f'expected at least {len(CAMERA_WRITE_METHODS)} camera-write member definitions '
            f'across the drivers, found {total_found}; CAMERA_WRITE_METHODS may be stale.'
        )

    def test_guard_flags_a_handler_that_falls_off_the_end(self):
        src = (
            'class Cam:\n'
            '    def gain(self, value):\n'
            '        try:\n'
            '            self.active.Gain.SetValue(value)\n'
            '        except Exception as e:\n'
            '            log(e)\n'
        )
        offenders, found = _swallowed_rejection_offenders(ast.parse(src))
        assert offenders and found == 1

    def test_guard_accepts_a_handler_that_reports_refused(self):
        src = (
            'class Cam:\n'
            '    def gain(self, value):\n'
            '        try:\n'
            '            self.active.Gain.SetValue(value)\n'
            '            return True\n'
            '        except Exception as e:\n'
            '            log(e)\n'
            '            return False\n'
        )
        offenders, found = _swallowed_rejection_offenders(ast.parse(src))
        assert not offenders and found == 1

    def test_guard_accepts_a_handler_that_reraises(self):
        src = (
            'class Cam:\n'
            '    def gain(self, value):\n'
            '        try:\n'
            '            self.active.Gain.SetValue(value)\n'
            '        except Exception:\n'
            '            raise\n'
        )
        offenders, found = _swallowed_rejection_offenders(ast.parse(src))
        assert not offenders and found == 1

    def test_guard_accepts_a_tolerating_handler_that_continues(self):
        """A handler with the member's real work after it is tolerating a step,
        not deciding the answer -- the GainSelector case."""
        src = (
            'class Cam:\n'
            '    def gain(self, value):\n'
            '        try:\n'
            '            self.active.GainSelector.SetValue("All")\n'
            '        except Exception as e:\n'
            '            log(e)\n'
            '        return True\n'
        )
        offenders, found = _swallowed_rejection_offenders(ast.parse(src))
        assert not offenders and found == 1

    def test_guard_is_not_fooled_by_an_annotation(self):
        """The rev 1 guard checked the return TYPE, which proves nothing."""
        src = (
            'class Cam:\n'
            '    def gain(self, value) -> bool:\n'
            '        try:\n'
            '            self.active.Gain.SetValue(value)\n'
            '        except Exception as e:\n'
            '            log(e)\n'
        )
        offenders, _found = _swallowed_rejection_offenders(ast.parse(src))
        assert offenders, 'an annotation is a declaration, not a behaviour'
