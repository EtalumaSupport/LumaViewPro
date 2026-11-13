# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Windows support report reads the host's CPU, memory, disks and display without `wmic`.

Current Windows 11 has no `wmic`, so a report that ran it carried
``[WinError 2]`` in all six fields. The report reads the same WMI classes
in-process, and each field holds its class's instances as ``Property=Value``
lines (a null as ``Property=``), or says why it has none: a class Windows
refuses, a class with no instances, or WMI that cannot be reached at all.
"""

import platform
import subprocess
import sys
import types

import pytest

from _lumascope.modules import tech_support_report

_FIELDS = ('cpu_detail', 'ram_detail', 'ram_total', 'disk_drives', 'disk_partitions', 'display')

_ANSWERS = {
    'Win32_Processor': [
        {
            'Name': 'Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz',
            'NumberOfCores': 4,
            'NumberOfLogicalProcessors': 8,
            'MaxClockSpeed': 2601,
        }
    ],
    'Win32_PhysicalMemory': [
        {'Capacity': '8589934592', 'Speed': 2133, 'Manufacturer': 'Samsung'},
        {'Capacity': '8589934592', 'Speed': 2133, 'Manufacturer': 'Samsung'},
    ],
    'Win32_ComputerSystem': [{'TotalPhysicalMemory': '17054613504'}],
    'Win32_DiskDrive': [
        {
            'Model': 'SAMSUNG MZVLW512',
            'Size': '512105932800',
            'InterfaceType': 'SCSI',
            'MediaType': 'Fixed hard disk media',
            'Status': 'OK',
        }
    ],
    'Win32_LogicalDisk': [
        {
            'DeviceID': 'C:',
            'Size': '511216201728',
            'FreeSpace': '201216201728',
            'FileSystem': 'NTFS',
            'VolumeName': None,
        }
    ],
    'Win32_VideoController': [
        {
            'Name': 'Intel(R) HD Graphics 530',
            'CurrentHorizontalResolution': 3840,
            'CurrentVerticalResolution': 2160,
            'CurrentRefreshRate': 59,
        }
    ],
}


class _ComError(Exception):
    pass


class _Wmi:
    """WMI's answer to a query: each class's instances, or the error Windows raises for it."""

    def __init__(self, answers):
        self._answers = answers

    def ExecQuery(self, query):
        answer = self._answers[query.rsplit(' FROM ', 1)[1]]
        if isinstance(answer, Exception):
            raise answer
        return [types.SimpleNamespace(**instance) for instance in answer]


@pytest.fixture
def ran(monkeypatch):
    """The host commands the report ran; each answers nothing."""
    # a stand-in by design: this host is not Windows, so its commands cannot run.
    commands = []

    def run(args, **kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout='', stderr='')

    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(tech_support_report.platform, 'system', lambda: 'Windows')
    return commands


def _windows_with_wmi(monkeypatch, answers):
    # a stand-in by design: this host has no Windows, so no COM and no WMI.
    client = types.ModuleType('win32com.client')
    client.GetObject = lambda moniker: _Wmi(answers)
    package = types.ModuleType('win32com')
    package.client = client
    monkeypatch.setitem(sys.modules, 'win32com', package)
    monkeypatch.setitem(sys.modules, 'win32com.client', client)
    monkeypatch.setitem(
        sys.modules,
        'pythoncom',
        types.SimpleNamespace(CoInitialize=lambda: None, CoUninitialize=lambda: None),
    )
    monkeypatch.setitem(sys.modules, 'pywintypes', types.SimpleNamespace(com_error=_ComError))


def test_each_field_is_its_classs_instances_and_no_wmic_runs(monkeypatch, ran):
    _windows_with_wmi(monkeypatch, _ANSWERS)

    info = tech_support_report._collect_system_info()

    assert not [args for args in ran if 'wmic' in args[0].lower()], ran
    assert info['cpu_detail'] == (
        'Name=Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz\n'
        'NumberOfCores=4\n'
        'NumberOfLogicalProcessors=8\n'
        'MaxClockSpeed=2601'
    )
    assert info['ram_detail'] == (
        'Capacity=8589934592\nSpeed=2133\nManufacturer=Samsung\n\n'
        'Capacity=8589934592\nSpeed=2133\nManufacturer=Samsung'
    )
    assert info['ram_total'] == 'TotalPhysicalMemory=17054613504'
    assert info['disk_drives'] == (
        'Model=SAMSUNG MZVLW512\nSize=512105932800\nInterfaceType=SCSI\n'
        'MediaType=Fixed hard disk media\nStatus=OK'
    )
    assert info['disk_partitions'] == (
        'DeviceID=C:\nSize=511216201728\nFreeSpace=201216201728\nFileSystem=NTFS\nVolumeName='
    )
    assert info['display'] == (
        'Name=Intel(R) HD Graphics 530\nCurrentHorizontalResolution=3840\n'
        'CurrentVerticalResolution=2160\nCurrentRefreshRate=59'
    )


def test_a_class_windows_refuses_says_so_and_the_others_are_read(monkeypatch, ran):
    _windows_with_wmi(monkeypatch, {**_ANSWERS, 'Win32_PhysicalMemory': _ComError('Access denied')})

    info = tech_support_report._collect_system_info()

    assert info['ram_detail'] == 'Win32_PhysicalMemory not read: Access denied'
    assert info['ram_total'] == 'TotalPhysicalMemory=17054613504'
    assert info['cpu_detail'].startswith('Name=Intel(R) Core(TM)')


def test_a_class_with_no_instances_says_so(monkeypatch, ran):
    _windows_with_wmi(monkeypatch, {**_ANSWERS, 'Win32_VideoController': []})

    info = tech_support_report._collect_system_info()

    assert info['display'] == 'Win32_VideoController: no instances'


def test_when_wmi_cannot_be_reached_every_field_says_why(monkeypatch, ran):
    # The module entry None makes its import raise, as on a host without pywin32.
    monkeypatch.setitem(sys.modules, 'win32com', None)
    monkeypatch.setitem(sys.modules, 'win32com.client', None)

    info = tech_support_report._collect_system_info()

    for field in _FIELDS:
        assert info[field].startswith('WMI not read: ModuleNotFoundError'), (field, info[field])


@pytest.mark.skipif(platform.system() != 'Windows', reason='reads the real host through WMI')
def test_on_a_windows_host_each_field_holds_its_first_property():
    info = tech_support_report._collect_system_info()

    first = {
        'cpu_detail': 'Name=',
        'ram_detail': 'Capacity=',
        'ram_total': 'TotalPhysicalMemory=',
        'disk_drives': 'Model=',
        'disk_partitions': 'DeviceID=',
        'display': 'Name=',
    }
    assert {f: info[f][: len(p)] for f, p in first.items()} == first, {f: info[f] for f in first}
