"""The microscope settings panel's camera lists are the API's answers.

The binning selector filled itself with [1, 2, 4] whenever the API's read
raised, which contradicts the API's own answer for no camera ([1]), and the
image-mode selector assumed 8-bit only. Neither read can raise for a reason
a correct program has: the API answers for an absent camera and every
driver contains its own read failure. So the panel adds nothing of its own,
and a raise is a defect that reaches the GUI boundary instead of a list.
"""

import types

import pytest

import modules.app_context as app_context


def _panel_with(monkeypatch, imaging):
    from ui.microscope_settings import MicroscopeSettings

    monkeypatch.setattr(
        app_context,
        'ctx',
        types.SimpleNamespace(
            lumaview=types.SimpleNamespace(scope=types.SimpleNamespace(imaging=imaging))
        ),
        raising=False,
    )
    spinners = {
        'binning_spinner': types.SimpleNamespace(values=None),
        'image_mode_spinner': types.SimpleNamespace(values=None),
    }
    return MicroscopeSettings, types.SimpleNamespace(ids=spinners)


def test_the_binning_list_is_exactly_the_apis(monkeypatch):
    imaging = types.SimpleNamespace(get_available_binning_sizes=lambda: [1, 2])
    cls, panel = _panel_with(monkeypatch, imaging)

    cls.load_binning_sizes(panel)

    assert panel.ids['binning_spinner'].values == ['1x1', '2x2']


def test_a_raise_from_the_binning_read_is_not_turned_into_a_list(monkeypatch):
    def _defect():
        raise RuntimeError('a defect')

    imaging = types.SimpleNamespace(get_available_binning_sizes=_defect)
    cls, panel = _panel_with(monkeypatch, imaging)

    with pytest.raises(RuntimeError):
        cls.load_binning_sizes(panel)
    assert panel.ids['binning_spinner'].values is None


def test_a_raise_from_the_pixel_format_read_is_not_turned_into_8_bit(monkeypatch):
    def _defect():
        raise RuntimeError('a defect')

    imaging = types.SimpleNamespace(get_supported_pixel_formats=_defect)
    cls, panel = _panel_with(monkeypatch, imaging)
    panel._supported_pixel_formats = lambda: cls._supported_pixel_formats(panel)

    with pytest.raises(RuntimeError):
        cls.load_image_modes(panel)
    assert panel.ids['image_mode_spinner'].values is None
