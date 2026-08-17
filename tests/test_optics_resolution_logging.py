"""Regression: the optics that set image scale are recorded in the log.

Every frame, hyperstack and still now carries a real PhysicalSizeX, but the
values that produce it -- tube-lens focal length and sensor pixel size off the
live scope, focal length off the selected objective -- were read and consumed
in process. Nothing wrote them down, so a returned support bundle could only
show the shipped default templates, which describe what ships rather than what
a given scope is set to. On a bench unit the defaults predicted 0.37657 um/px
while images were actually written at 0.37744, and the bundle could not say
why.

The logged um/px must come from the same resolver that feeds the images. Two
implementations of the scale formula would make the logged number worthless as
evidence, which is the whole point of recording it.
"""

import pathlib
from unittest.mock import MagicMock, patch

import pytest

import modules.app_context as _app_ctx
import modules.common_utils as common_utils


TUBE_FOCAL_LENGTH_MM = 47.8
SENSOR_PIXEL_SIZE_UM = 2.0
OBJECTIVE_FOCAL_LENGTH_MM = 9.0


@pytest.fixture
def scope_with_optics():
    """A scope that can report both optics values."""
    ctx = MagicMock()
    ctx.scope.capabilities.lens_focal_length_mm = TUBE_FOCAL_LENGTH_MM
    ctx.scope.capabilities.pixel_size_um = SENSOR_PIXEL_SIZE_UM
    original = _app_ctx.ctx
    _app_ctx.ctx = ctx
    try:
        yield ctx
    finally:
        _app_ctx.ctx = original


@pytest.fixture
def scope_without_optics():
    """A scope that cannot report its optics -- unknown camera, no declared
    optics. The resolver returns None here rather than inventing a scale."""
    ctx = MagicMock()
    ctx.scope.capabilities.lens_focal_length_mm = None
    ctx.scope.capabilities.pixel_size_um = SENSOR_PIXEL_SIZE_UM
    original = _app_ctx.ctx
    _app_ctx.ctx = ctx
    try:
        yield ctx
    finally:
        _app_ctx.ctx = original


def _emitted(mock_logger):
    """Every message the call put on the log, whatever the level."""
    return [str(call.args[0]) for call in mock_logger.method_calls if call.args]


class TestLoggedScaleCannotDriftFromWrittenScale:
    def test_logged_um_per_pixel_is_the_resolver_s_own_answer(self, scope_with_optics):
        """The number logged and the number baked into images are one value."""
        expected = common_utils.get_pixel_size(
            focal_length=OBJECTIVE_FOCAL_LENGTH_MM, binning_size=1
        )

        with patch.object(common_utils, 'logger') as mock_logger:
            common_utils.log_resolved_optics(
                objective_id='20x', focal_length=OBJECTIVE_FOCAL_LENGTH_MM, binning_size=1
            )

        line = ' '.join(_emitted(mock_logger))
        assert f'{expected}um/px' in line, (
            f'logged scale must be the resolver output {expected}; got: {line}'
        )

    def test_every_input_to_the_scale_is_on_the_line(self, scope_with_optics):
        """A bundle has to be able to tell a misconfigured scope from a wrong
        resolver, which needs the inputs and not only the result."""
        with patch.object(common_utils, 'logger') as mock_logger:
            common_utils.log_resolved_optics(
                objective_id='20x', focal_length=OBJECTIVE_FOCAL_LENGTH_MM, binning_size=2
            )

        line = ' '.join(_emitted(mock_logger))
        for expected in (
            '20x',
            str(OBJECTIVE_FOCAL_LENGTH_MM),
            str(TUBE_FOCAL_LENGTH_MM),
            str(SENSOR_PIXEL_SIZE_UM),
            'binning=2',
        ):
            assert expected in line, f'{expected!r} missing from the optics line: {line}'

    def test_binning_reaches_the_resolver(self, scope_with_optics):
        """Binning multiplies the scale; logging it unbinned would mislabel
        every binned image in the bundle."""
        with patch.object(common_utils, 'logger') as mock_logger:
            common_utils.log_resolved_optics(
                objective_id='20x', focal_length=OBJECTIVE_FOCAL_LENGTH_MM, binning_size=2
            )

        unbinned = common_utils.get_pixel_size(
            focal_length=OBJECTIVE_FOCAL_LENGTH_MM, binning_size=1
        )
        line = ' '.join(_emitted(mock_logger))
        assert f'{unbinned * 2}um/px' in line


class TestNoScaleIsItselfReported:
    def test_missing_optics_still_logs_and_names_what_is_missing(self, scope_without_optics):
        """The condition a returned bundle most needs explained is the one
        where there is no scale at all -- staying silent there is the defect."""
        with patch.object(common_utils, 'logger') as mock_logger:
            common_utils.log_resolved_optics(
                objective_id='20x', focal_length=OBJECTIVE_FOCAL_LENGTH_MM, binning_size=1
            )

        line = ' '.join(_emitted(mock_logger))
        assert line, 'a scope that cannot report its optics logged nothing at all'
        assert 'tube lens focal length' in line, (
            f'the missing input must be named, not printed as None: {line}'
        )
        assert 'None' not in line


class TestResolverStaysSilent:
    def test_get_pixel_size_does_not_log(self):
        """Structural: the resolver runs on the scale-bar and field-of-view
        paths, far more often than the optics change. Logging inside it would
        bury the event it is meant to record.
        """
        source = (
            pathlib.Path(__file__).resolve().parent.parent / 'modules' / 'common_utils.py'
        ).read_text()
        start = source.index('def get_pixel_size(')
        body = source[start : source.index('\ndef ', start)]
        assert 'logger.' not in body, (
            'get_pixel_size logs. It is called per-frame by the scale bar and '
            'field-of-view paths; the optics record belongs at the point the '
            'optics are chosen, not at every read.'
        )
