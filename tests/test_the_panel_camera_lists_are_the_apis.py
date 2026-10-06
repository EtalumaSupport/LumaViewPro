"""The microscope settings panel's camera lists are the API's answers.

The binning selector filled itself with [1, 2, 4] whenever the API's read
raised, which contradicts the API's own answer for no camera, and the
image-mode selector assumed 8-bit only. The binning sizes are a static fact
of the camera, read once at connect into ``scope.capabilities``; the panel
shows that list and adds nothing of its own.
"""

import types

import pytest

import modules.app_context as app_context


def _panel_with(monkeypatch, binning_sizes):
    from ui.microscope_settings import MicroscopeSettings

    scope = types.SimpleNamespace(
        capabilities=types.SimpleNamespace(camera_binning_sizes=binning_sizes)
    )
    monkeypatch.setattr(
        app_context,
        'ctx',
        types.SimpleNamespace(lumaview=types.SimpleNamespace(scope=scope)),
        raising=False,
    )
    spinners = {
        'binning_spinner': types.SimpleNamespace(values=None),
        'image_mode_spinner': types.SimpleNamespace(values=None),
    }
    return MicroscopeSettings, types.SimpleNamespace(ids=spinners)


def test_the_binning_list_is_exactly_the_capabilities(monkeypatch):
    cls, panel = _panel_with(monkeypatch, (1, 2))

    cls.load_binning_sizes(panel)

    assert panel.ids['binning_spinner'].values == ['1x1', '2x2']


def test_no_camera_offers_no_binning(monkeypatch):
    cls, panel = _panel_with(monkeypatch, ())

    cls.load_binning_sizes(panel)

    assert panel.ids['binning_spinner'].values == []


def test_a_camera_profile_defect_is_not_turned_into_one_size(monkeypatch):
    """Behind the panel the API answered [1] for a profile it could not read.

    Every camera has a profile (`Camera.__init__`, `lookup_profile`), so the
    only way to reach that answer was a defect, which it hid as a camera that
    cannot bin.
    """
    from modules.lumascope_api.imaging import ImagingAPI

    class _NoSizes:
        def __getattr__(self, name):
            raise AttributeError(name)

    camera = types.SimpleNamespace(active=True, profile=_NoSizes())
    monkeypatch.setattr(ImagingAPI, '_driver', property(lambda self: camera))
    api = ImagingAPI.__new__(ImagingAPI)

    with pytest.raises(AttributeError):
        api.get_available_binning_sizes()
