# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The binning in force has one answerer, the Session.

The GUI read it through a ``config_ui_getters`` forwarder of its own, so a
script had no read and the GUI's pixel-size and field-of-view readouts
had a second door onto the store. ``ScopeSession.get_binning_size`` is the
one: it answers the factor ``set_binning_size`` stored, which is stored
only once the camera took it.
"""

from __future__ import annotations

import logging

import pytest

from modules.exceptions import (
    CameraSettingUnsupportedError,
    HardwareCommandRefusedError,
    MissingPart,
)
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session():
    built = ScopeSession.create(complete_settings(), simulate=True)
    yield built
    try:
        built.shutdown()
    except Exception:
        logging.getLogger(__name__).debug('teardown noise is not the measurement', exc_info=True)


def test_it_answers_the_binning_the_camera_took(session):
    offered = session.scope.capabilities.camera_binning_sizes
    size = max(offered)

    session.set_binning_size(size)

    assert session.get_binning_size() == size


def test_a_refused_binning_leaves_the_answer_as_it_was(session):
    before = session.get_binning_size()
    unsupported = max(session.scope.capabilities.camera_binning_sizes) * 16

    with pytest.raises(CameraSettingUnsupportedError):
        session.set_binning_size(unsupported)

    assert session.get_binning_size() == before


def test_with_no_camera_nothing_is_binned_and_nothing_is_stored(session, monkeypatch):
    before = session.get_binning_size()
    monkeypatch.setattr(type(session.scope), 'camera_connected', property(lambda self: False))

    with pytest.raises(HardwareCommandRefusedError) as exc:
        session.set_binning_size(1)
    assert exc.value.missing == MissingPart.CAMERA

    assert session.get_binning_size() == before
