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

import ast
import logging
from types import SimpleNamespace

import pytest

import modules.common_utils as common_utils
import modules.config_helpers as config_helpers
from tests.ast_seams import find_def
from tests.log_capture import messages, records_of


TUBE_FOCAL_LENGTH_MM = 47.8
SENSOR_PIXEL_SIZE_UM = 2.0
OBJECTIVE_FOCAL_LENGTH_MM = 9.0


@pytest.fixture
def optics():
    """The capabilities of a scope that can report both optics values --
    handed to the record the way the Session hands in its scope's."""
    return SimpleNamespace(
        lens_focal_length_mm=TUBE_FOCAL_LENGTH_MM, pixel_size_um=SENSOR_PIXEL_SIZE_UM
    )


@pytest.fixture
def no_optics():
    """A scope that cannot report its optics -- unknown camera, no declared
    optics. The resolver returns None here rather than inventing a scale."""
    return SimpleNamespace(lens_focal_length_mm=None, pixel_size_um=SENSOR_PIXEL_SIZE_UM)


def _emitted(caplog):
    """Every message config_helpers put on the log, whatever the level."""
    return messages(records_of(caplog, config_helpers))


class TestLoggedScaleCannotDriftFromWrittenScale:
    def test_logged_um_per_pixel_is_the_resolver_s_own_answer(self, optics, caplog):
        """The number logged and the number baked into images are one value."""
        expected = common_utils.get_pixel_size(
            focal_length=OBJECTIVE_FOCAL_LENGTH_MM,
            binning_size=1,
            capabilities=optics,
        )

        with caplog.at_level(logging.DEBUG):
            config_helpers.log_resolved_optics(
                objective_id='20x',
                focal_length=OBJECTIVE_FOCAL_LENGTH_MM,
                binning_size=1,
                capabilities=optics,
            )

        line = ' '.join(_emitted(caplog))
        assert f'{expected}um/px' in line, (
            f'logged scale must be the resolver output {expected}; got: {line}'
        )

    def test_every_input_to_the_scale_is_on_the_line(self, optics, caplog):
        """A bundle has to be able to tell a misconfigured scope from a wrong
        resolver, which needs the inputs and not only the result."""
        with caplog.at_level(logging.DEBUG):
            config_helpers.log_resolved_optics(
                objective_id='20x',
                focal_length=OBJECTIVE_FOCAL_LENGTH_MM,
                binning_size=2,
                capabilities=optics,
            )

        line = ' '.join(_emitted(caplog))
        for expected in (
            '20x',
            str(OBJECTIVE_FOCAL_LENGTH_MM),
            str(TUBE_FOCAL_LENGTH_MM),
            str(SENSOR_PIXEL_SIZE_UM),
            'binning=2',
        ):
            assert expected in line, f'{expected!r} missing from the optics line: {line}'

    def test_binning_reaches_the_resolver(self, optics, caplog):
        """Binning multiplies the scale; logging it unbinned would mislabel
        every binned image in the bundle."""
        with caplog.at_level(logging.DEBUG):
            config_helpers.log_resolved_optics(
                objective_id='20x',
                focal_length=OBJECTIVE_FOCAL_LENGTH_MM,
                binning_size=2,
                capabilities=optics,
            )

        unbinned = common_utils.get_pixel_size(
            focal_length=OBJECTIVE_FOCAL_LENGTH_MM,
            binning_size=1,
            capabilities=optics,
        )
        line = ' '.join(_emitted(caplog))
        assert f'{unbinned * 2}um/px' in line


class TestNoScaleIsItselfReported:
    def test_missing_optics_still_logs_and_names_what_is_missing(self, no_optics, caplog):
        """The condition a returned bundle most needs explained is the one
        where there is no scale at all -- staying silent there is the defect."""
        with caplog.at_level(logging.DEBUG):
            config_helpers.log_resolved_optics(
                objective_id='20x',
                focal_length=OBJECTIVE_FOCAL_LENGTH_MM,
                binning_size=1,
                capabilities=no_optics,
            )

        line = ' '.join(_emitted(caplog))
        assert line, 'a scope that cannot report its optics logged nothing at all'
        assert 'tube lens focal length' in line, (
            f'the missing input must be named, not printed as None: {line}'
        )
        assert 'None' not in line

    def test_each_input_the_resolver_lacked_is_named(self, optics, caplog):
        """Every way the resolver can answer no scale is named on the line,
        so the missing list is never empty: the binning, read off the
        camera, is unknown until the camera reports it, and a bundle showed
        "missing: ." for exactly that case."""
        cases = {
            'active scope': (1, None),
            'tube lens focal length': (
                1,
                SimpleNamespace(lens_focal_length_mm=None, pixel_size_um=SENSOR_PIXEL_SIZE_UM),
            ),
            'sensor pixel size': (
                1,
                SimpleNamespace(lens_focal_length_mm=TUBE_FOCAL_LENGTH_MM, pixel_size_um=None),
            ),
            'camera binning': (None, optics),
        }
        unnamed = {}
        for expected, (binning_size, capabilities) in cases.items():
            caplog.clear()
            with caplog.at_level(logging.DEBUG):
                config_helpers.log_resolved_optics(
                    objective_id='20x',
                    focal_length=OBJECTIVE_FOCAL_LENGTH_MM,
                    binning_size=binning_size,
                    capabilities=capabilities,
                )
            line = ' '.join(_emitted(caplog))
            named = line.partition('missing: ')[2].partition('.')[0]
            if expected not in named:
                unnamed[expected] = line
        assert not unnamed, f'a no-scale line did not name what was missing: {unnamed}'


class TestResolverStaysSilent:
    def test_get_pixel_size_does_not_log(self):
        """Structural: the resolver runs on the scale-bar and field-of-view
        paths, far more often than the optics change. Logging inside it would
        bury the event it is meant to record.
        """
        resolver = find_def('modules/common_utils.py', 'get_pixel_size')
        offenders = [
            node.lineno
            for node in ast.walk(resolver)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'logger'
        ]
        assert not offenders, (
            f'get_pixel_size logs at line(s) {offenders}. It is called per frame '
            'by the scale bar and field-of-view paths; the optics record belongs '
            'at the point the optics are chosen, not at every read.'
        )
