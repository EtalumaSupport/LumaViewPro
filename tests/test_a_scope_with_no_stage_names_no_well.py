# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A scope with no XY stage names no well.

The well label was read from the X and Y targets one axis at a time, and a
single-axis read of an axis the scope does not have answers 0 -- so a
still on an LS820 was named after whatever well sits at plate (0, 0), and
the file claimed a place the scope cannot know. When the label was None,
the still told the user to home the scope, which gives a scope with no
stage no position either.

Driven through the Session on the simulator: an LS820 (Z only) against an
LS850 (X, Y and Z) as the control.
"""

import json

import pytest
import tifffile

from tests.test_manual_capture_member import _capture, _open_session, _settings


def _session(tmp_path, microscope):
    settings = _settings(tmp_path)
    settings['microscope'] = microscope
    return _open_session(settings)


@pytest.fixture
def warnings_shown(monkeypatch):
    from modules.notification_center import notifications

    shown = []
    monkeypatch.setattr(
        notifications,
        'warning',
        lambda category, title, message, **kwargs: shown.append(title),
    )
    return shown


def _description(path):
    with tifffile.TiffFile(str(path)) as tf:
        return json.loads(tf.pages[0].tags['ImageDescription'].value)


class TestAZOnlyScope:
    def test_the_well_label_is_none(self, tmp_path):
        with _session(tmp_path, 'LS820') as session:
            assert not session.scope.capabilities.has_xy_stage
            assert session.scope.runtime_state.get_well_label() is None

    def test_a_still_is_saved_without_a_well_and_says_nothing(self, tmp_path, warnings_shown):
        with _session(tmp_path, 'LS820') as session:
            (path,) = _capture(session)

        assert path.name == 'live_BF_000001.tiff'
        assert _description(path)['Plate']['WellLabel'] == ''
        assert warnings_shown == []


class TestAnXYScope:
    def test_a_homed_stage_names_its_well(self, tmp_path, warnings_shown):
        with _session(tmp_path, 'LS850') as session:
            assert session.scope.capabilities.has_xy_stage
            label = session.scope.runtime_state.get_well_label()
            (path,) = _capture(session)

        assert label
        assert path.name == f'live_{label}_BF_000001.tiff'
        assert _description(path)['Plate']['WellLabel'] == label
        assert warnings_shown == []
