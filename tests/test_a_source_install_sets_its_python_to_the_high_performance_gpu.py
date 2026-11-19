# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A source install sets its Python to Windows' High performance GPU, keeping the setting's other keys.

LumaViewPro run from source draws on the GPU Windows gives its Python; a
laptop's Intel drew the live view at 12 fps where its NVIDIA drew 42.
``install_windows.bat`` runs ``scripts/install_gpu_preference.py``, which
writes ``GpuPreference=2`` into the interpreter's setting. The setting is
one string of ``key=value;`` pairs Windows' Graphics page may hold other
keys in, so only that key is written, and a choice it replaced is named.
The registry write itself is Windows'; its value is what is pinned here.
"""

from __future__ import annotations

import importlib.util
import pathlib

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'install_gpu_preference.py'
_spec = importlib.util.spec_from_file_location('install_gpu_preference', _SCRIPT)
install_gpu_preference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(install_gpu_preference)
with_high_performance = install_gpu_preference.with_high_performance


def test_an_unset_python_is_set_to_high_performance():
    assert with_high_performance(None) == ('GpuPreference=2;', None)


def test_the_settings_other_keys_are_kept():
    stored, replaced = with_high_performance('SwapEffectUpgradeEnable=1;AutoHDREnable=0;')

    assert stored == 'SwapEffectUpgradeEnable=1;AutoHDREnable=0;GpuPreference=2;'
    assert replaced is None


def test_a_power_saving_choice_is_replaced_and_named():
    stored, replaced = with_high_performance('GpuPreference=1;SwapEffectUpgradeEnable=1;')

    assert stored == 'SwapEffectUpgradeEnable=1;GpuPreference=2;'
    assert replaced == '1'


def test_a_python_already_at_high_performance_names_no_replacement():
    assert with_high_performance('GpuPreference=2;') == ('GpuPreference=2;', None)
