"""The frame-validity chunk target must describe the exposure the hardware
actually applied, not the one that was requested.

The bench defect this locks: a stored 0.01 ms exposure was requested as 10 us,
the driver clamped it up to the node minimum (~14 us) and reported nothing, and
the API stamped the REQUEST as the chunk-match target. Every frame then arrived
carrying the applied value, missed the 2.0 us tolerance by 4 us, and was
rejected -- ten captures in a row, with no value the camera would ever produce
that could close the gate.

The clamp is correct and stays. What must hold is that the target follows the
applied value through it.
"""

import sys
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault('modules.settings_init', MagicMock())


CHUNK_TOLERANCE_US = 2.0


class _RecordingValidity:
    """Minimal frame-validity stand-in that records targets by source."""

    def __init__(self):
        self.targets = {}

    def invalidate(self, source):
        pass

    def set_target(self, source, value):
        self.targets[source] = value

    def chunk_match(self, source, chunk_value):
        target = self.targets.get(source)
        if target is None or chunk_value is None:
            return False
        return abs(float(chunk_value) - target) <= CHUNK_TOLERANCE_US


def _imaging_with(result):
    """An imaging object whose driver write returns ``result``."""
    from modules.lumascope_api.imaging import ImagingAPI

    obj = ImagingAPI.__new__(ImagingAPI)
    obj.frame_validity = _RecordingValidity()
    obj._commit_camera_writes = lambda updates: None
    return obj, lambda: result


def test_clamped_write_targets_the_applied_value_not_the_request():
    """The bench case: request 10 us, driver applies 14 us, frame reports 14."""
    imaging, write_fn = _imaging_with(14.0)

    imaging._camera_write(
        write_fn,
        targets=(('exposure', 10.0),),
        target_from_result=('exposure',),
    )

    assert imaging.frame_validity.targets['exposure'] == 14.0, (
        'the chunk target must be the applied microseconds; recording the '
        'request is what rejected every frame at the bench'
    )
    assert imaging.frame_validity.chunk_match('exposure', 14.0), (
        'a frame carrying the applied exposure must pass the gate'
    )


def test_short_circuited_write_still_reports_the_value_in_effect():
    """No write occurs, but the clamped value is still what the camera uses.

    This is the live path at every bench rejection -- the driver short-circuits
    because the node already holds the clamped value, so a driver that returned
    None here would leave the stale request standing as the target.
    """
    imaging, write_fn = _imaging_with(14.0)

    imaging._camera_write(
        write_fn,
        targets=(('exposure', 10.0),),
        target_from_result=('exposure',),
    )

    assert imaging.frame_validity.targets['exposure'] == 14.0


def test_refused_write_records_no_target():
    """A driver that refused did not move the hardware."""
    imaging, write_fn = _imaging_with(False)

    imaging._camera_write(
        write_fn,
        targets=(('exposure', 10.0),),
        target_from_result=('exposure',),
    )

    assert 'exposure' not in imaging.frame_validity.targets, (
        'a refused write must not claim a target the camera never took'
    )


def test_driver_reporting_no_value_falls_back_to_the_request():
    """None means applied-but-unknown; the request is the best available target."""
    imaging, write_fn = _imaging_with(None)

    imaging._camera_write(
        write_fn,
        targets=(('exposure', 10.0),),
        target_from_result=('exposure',),
    )

    assert imaging.frame_validity.targets['exposure'] == 10.0


def test_a_bare_true_is_not_mistaken_for_microseconds():
    """bool is an int subclass; a driver reporting True must not stamp 1.0 us."""
    imaging, write_fn = _imaging_with(True)

    imaging._camera_write(
        write_fn,
        targets=(('exposure', 10.0),),
        target_from_result=('exposure',),
    )

    assert imaging.frame_validity.targets['exposure'] == 10.0, (
        'True is applied-but-unknown, not a 1 microsecond exposure'
    )


@pytest.mark.parametrize(
    'driver_module,driver_name',
    [
        ('drivers.pyloncamera', 'PylonCamera'),
        ('drivers.idscamera', 'IDSCamera'),
        ('drivers.fx2driver', 'FX2Camera'),
        ('drivers.simulated_camera', 'SimulatedCamera'),
    ],
)
def test_every_driver_declares_the_applied_value_contract(driver_module, driver_name):
    """One contract across all four drivers, so it cannot drift back apart.

    The defect was possible because the same method had four different return
    types; an annotation of ``None`` here means a transforming driver has no way
    to report what it applied.
    """
    import importlib
    import inspect

    mod = importlib.import_module(driver_module)
    cls = getattr(mod, driver_name, None)
    if cls is None:
        pytest.skip(f'{driver_name} not importable in this environment')

    sig = inspect.signature(cls.exposure_t)
    annotation = sig.return_annotation
    assert annotation is not None and str(annotation) != 'None', (
        f'{driver_name}.exposure_t must be able to report the applied value'
    )
    assert 'float' in str(annotation), (
        f'{driver_name}.exposure_t returns {annotation!r}; it must be able to '
        'report applied microseconds so the chunk target can follow it'
    )
