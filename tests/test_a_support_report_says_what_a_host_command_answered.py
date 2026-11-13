# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A support report says what each host command answered, an empty answer included.

The report reads the host through its own commands (``system_profiler``,
``sysctl``, ``lsusb``, ``powercfg``). A command that prints
nothing -- a ``system_profiler`` data type the OS no longer has, a refusal
written only to stderr -- is said in the report with its exit status and its
stderr, never left as an empty file or field: on macOS 26 the report's USB
listing asked for ``SPUSBDataType``, which that OS no longer has, and every
Mac report carried an empty ``SPUSBDataType.txt``.
"""

import platform
import sys

import pytest

from _lumascope.modules import tech_support_report


def test_a_command_is_reported_by_its_output_or_by_why_it_has_none():
    # Real commands, the interpreter running this test: one that prints, one
    # that prints nothing, one that fails with only stderr to say why.
    printed = tech_support_report._run([sys.executable, '-c', 'print("  bus 1  ")'])
    silent = tech_support_report._run([sys.executable, '-c', 'pass'])
    refused = tech_support_report._run(
        [sys.executable, '-c', 'import sys; sys.stderr.write("no such type"); sys.exit(3)']
    )

    assert printed == 'bus 1'
    assert silent.endswith('answered nothing (exit 0)')
    assert refused.endswith('answered nothing (exit 3): no such type')


@pytest.mark.skipif(platform.system() != 'Darwin', reason='the USB listing a Mac report asks for')
def test_a_mac_report_lists_the_usb_devices_this_macos_answers_with(monkeypatch):
    # a stand-in by design: the host's serial ports, which no test may touch.
    monkeypatch.setattr('serial.tools.list_ports.comports', lambda include_links: [])

    usb = dict(tech_support_report._collect_usb_devices())

    assert usb['SPUSBHostDataType'].startswith('USB:'), usb['SPUSBHostDataType'][:200]
    assert all(usb[t].strip() for t in ('SPUSBHostDataType', 'SPUSBDataType')), {
        t: usb[t][:200] for t in ('SPUSBHostDataType', 'SPUSBDataType')
    }
