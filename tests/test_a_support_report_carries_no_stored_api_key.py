# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A support report copies the data folder with every stored ``api_key`` value blanked.

The data folder's copy (``TechSupportReport._step_data_folder``, both
reports' one copier) blanks the value of every ``"api_key"`` in every
``current.json*`` -- the live file, each file set aside as rejected, a
``.saving`` a power cut left -- and keeps every other byte, on the bytes:
those files are copied because they may not parse, and a rejected one may
not decode, so support can read the corruption. A copy that fails fails the
report; a file the settings saver renames away between the listing and its
copy is not there to copy, and is left out and named.

Driven through ``make_logs_zip``, the cheaper of the two reports that share
the copier.
"""

from __future__ import annotations

import os
import pathlib
import sys
import zipfile

import pytest

from modules import tech_support_report
from modules.exceptions import SupportReportNotSavedError

# Each file as it sits in the data folder, and as the report must carry it.
_FILES = {
    # The live file, in the platform encoding (an accented path, cp1252),
    # with a plugin's own key beside the server's.
    'current.json': (
        b'{"live_folder": "C:\\\\Users\\\\Ren\xe9", "plugins": {"x": {"api_key": "plugin-own"}},'
        b' "rest_api": {"enabled": false, "port": 8000, "api_key": "s3cr3t"}}',
        b'{"live_folder": "C:\\\\Users\\\\Ren\xe9", "plugins": {"x": {"api_key": ""}},'
        b' "rest_api": {"enabled": false, "port": 8000, "api_key": ""}}',
    ),
    # Set aside because it did not parse: not JSON, nor decodable.
    'current.json.rejected-20261021-101500': (
        b'{"rest_api": {"api_key": "s3cr3t", "port": 8000,,, \xff\xfe',
        b'{"rest_api": {"api_key": "", "port": 8000,,, \xff\xfe',
    ),
    # A power cut mid-write: cut off inside the value, then a NUL tail.
    'current.json.saving': (
        b'{\n    "rest_api": {\n        "api_key": "s3cr' + b'\x00' * 8,
        b'{\n    "rest_api": {\n        "api_key": ""',
    ),
    # Shipped null: kept as it is.
    'current.json.rejected-20261020-090000': (
        b'{"rest_api": {"api_key": null}}',
        b'{"rest_api": {"api_key": null}}',
    ),
    # Not a settings file: copied as it is.
    'notes/api_key.txt': (b'"api_key": "not a setting"', b'"api_key": "not a setting"'),
}


@pytest.fixture
def data(tmp_path, monkeypatch):
    root = tmp_path / 'installation'
    folder = root / 'data'
    for name, (stored, _) in _FILES.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(stored)
    # a stand-in by design: the installation's folder, which the report
    # finds from where LumaViewPro runs.
    monkeypatch.setattr(tech_support_report, 'get_source_root', lambda *args, **kwargs: root)
    return folder


@pytest.fixture
def session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    live = tmp_path / 'live'
    live.mkdir()
    session = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    yield session
    session.shutdown()


def _copied(zip_path: pathlib.Path) -> dict[str, bytes]:
    with zipfile.ZipFile(zip_path) as z:
        return {
            name.partition('data/')[2]: z.read(name)
            for name in z.namelist()
            if '/data/' in f'/{name}' and not name.endswith('/')
        }


def test_every_stored_api_key_is_blanked_and_every_other_byte_kept(session, data, tmp_path):
    copied = _copied(session.make_logs_zip(output_dir=tmp_path / 'out').path)

    assert copied == {name: carried for name, (_, carried) in _FILES.items()}


@pytest.mark.skipif(
    sys.platform == 'win32' or os.geteuid() == 0, reason='a file its reader may not open'
)
def test_a_copy_that_fails_fails_the_report(session, data, tmp_path):
    unreadable = data / 'current.json'
    unreadable.chmod(0)
    try:
        with pytest.raises(SupportReportNotSavedError) as raised:
            session.make_logs_zip(output_dir=tmp_path / 'out')
    finally:
        unreadable.chmod(0o644)

    assert 'current.json' in str(raised.value)
    assert not list(tmp_path.rglob('*.zip'))


class _Listed:
    def __init__(self, entries):
        self._entries = entries

    def __enter__(self):
        return iter(self._entries)

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._entries)


def test_a_saving_file_renamed_away_mid_copy_is_left_out_and_named(
    session, data, tmp_path, monkeypatch
):
    real_scandir = os.scandir

    def listed_then_saved(path='.'):
        if isinstance(path, int) or pathlib.Path(path) != data:
            return real_scandir(path)
        with real_scandir(path) as listing:
            entries = list(listing)
        if (data / 'current.json.saving').exists():
            # The saver's rename, landing between the listing and the copy.
            os.replace(data / 'current.json.saving', data / 'current.json')
        return _Listed(entries)

    # a stand-in by design: times the settings saver's os.replace into the
    # copier's walk, a window no simulated scope can be timed to hit.
    monkeypatch.setattr(os, 'scandir', listed_then_saved)
    copied = _copied(session.make_logs_zip(output_dir=tmp_path / 'out').path)

    assert 'current.json.saving' not in copied
    # What the rename put in place, blanked.
    assert copied['current.json'] == _FILES['current.json.saving'][1]
    assert b'current.json.saving' in copied['NOT_COPIED.txt']
