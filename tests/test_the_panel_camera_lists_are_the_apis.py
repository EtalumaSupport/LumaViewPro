"""The microscope settings panel's camera lists are the API's answers.

The binning selector filled itself with [1, 2, 4] whenever the API's read
raised, which contradicts the API's own answer for no camera, and the
image-mode selector assumed 8-bit only. The binning sizes are a static fact
of the camera, read once at connect into ``scope.capabilities``; the panel
shows that list and adds nothing of its own.
"""

import types

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
