"""The installer's own logs must reach the application log folder.

The Windows bundle writes its logs to the user TEMP directory, where a
support bundle never sees them -- so an install that silently failed to
replace a binary reads exactly like an application defect, and Windows
sweeps the evidence away on its own schedule. Startup copies them into
the app's log folder instead. These exercise the real function; the
temp-directory argument makes the copy observable without touching the
machine's actual TEMP.
"""

from __future__ import annotations

import pathlib

from modules.app_environment import capture_installer_logs


def _make(directory: pathlib.Path, name: str, content: str = 'log body') -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(content, encoding='utf-8')
    return path


def test_matching_installer_logs_are_copied_into_the_install_subfolder(tmp_path):
    temp_dir = tmp_path / 'temp'
    log_dir = tmp_path / 'logs'
    _make(temp_dir, 'LumaViewPro-4.0.0-beta23_20260730150620.log')
    _make(temp_dir, 'LumaViewPro-4.0.0-beta23_20260730150620_0_LVP.log')

    copied = capture_installer_logs(log_dir, temp_dir=temp_dir)

    assert sorted(copied) == [
        'LumaViewPro-4.0.0-beta23_20260730150620.log',
        'LumaViewPro-4.0.0-beta23_20260730150620_0_LVP.log',
    ]
    for name in copied:
        assert (log_dir / 'install' / name).read_text(encoding='utf-8') == 'log body'


def test_unrelated_temp_files_are_left_alone(tmp_path):
    temp_dir = tmp_path / 'temp'
    log_dir = tmp_path / 'logs'
    _make(temp_dir, 'someone_elses_installer.log')
    _make(temp_dir, 'LumaViewPro-notes.txt')

    assert capture_installer_logs(log_dir, temp_dir=temp_dir) == []
    assert not (log_dir / 'install').exists()


def test_repeat_startup_does_not_recopy_captured_logs(tmp_path):
    temp_dir = tmp_path / 'temp'
    log_dir = tmp_path / 'logs'
    _make(temp_dir, 'LumaViewPro-4.0.0-beta23_1.log')

    first = capture_installer_logs(log_dir, temp_dir=temp_dir)
    second = capture_installer_logs(log_dir, temp_dir=temp_dir)

    assert first == ['LumaViewPro-4.0.0-beta23_1.log']
    assert second == [], 'an already-captured log must not be copied again'


def test_a_grown_log_is_recaptured(tmp_path):
    temp_dir = tmp_path / 'temp'
    log_dir = tmp_path / 'logs'
    source = _make(temp_dir, 'LumaViewPro-4.0.0-beta23_2.log', content='partial')
    capture_installer_logs(log_dir, temp_dir=temp_dir)

    source.write_text('partial plus the rest of the install', encoding='utf-8')
    again = capture_installer_logs(log_dir, temp_dir=temp_dir)

    assert again == ['LumaViewPro-4.0.0-beta23_2.log']
    assert (log_dir / 'install' / 'LumaViewPro-4.0.0-beta23_2.log').read_text(
        encoding='utf-8'
    ) == 'partial plus the rest of the install'


def test_missing_temp_directory_is_not_fatal(tmp_path):
    assert capture_installer_logs(tmp_path / 'logs', temp_dir=tmp_path / 'nope') == []


def test_only_the_newest_logs_are_copied(tmp_path):
    temp_dir = tmp_path / 'temp'
    log_dir = tmp_path / 'logs'
    for index in range(5):
        path = _make(temp_dir, f'LumaViewPro-4.0.0-beta23_{index}.log')
        import os

        os.utime(path, (1_700_000_000 + index, 1_700_000_000 + index))

    copied = capture_installer_logs(log_dir, temp_dir=temp_dir, max_files=2)

    assert sorted(copied) == [
        'LumaViewPro-4.0.0-beta23_3.log',
        'LumaViewPro-4.0.0-beta23_4.log',
    ]


def test_startup_captures_installer_logs_after_logging_is_available():
    source = (pathlib.Path(__file__).resolve().parents[1] / 'lumaviewpro.py').read_text()
    call = source.find('capture_installer_logs(')
    logger_import = source.find('from lvp_logger import')
    assert call != -1, 'lumaviewpro.py must capture installer logs at startup'
    assert logger_import != -1
    assert logger_import < call, (
        'the capture must run after the logger exists so its own failures are recorded'
    )


def _make_utf16(directory: pathlib.Path, name: str, content: str) -> pathlib.Path:
    """Write a file the way MSI writes its verbose log: UTF-16 with a BOM."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b'\xff\xfe' + content.encode('utf-16-le'))
    return path


def test_a_utf16_installer_log_lands_as_utf8(tmp_path):
    """MSI's verbose log is UTF-16, which costs two bytes per character for
    no added information. These sit in the user's Documents folder and are
    never pruned, so the copy transcodes instead of storing the wide form."""
    temp_dir = tmp_path / 'temp'
    log_dir = tmp_path / 'logs'
    body = '=== Verbose logging started ===\r\nExecuting op: FileCopy(...)\r\n'
    source = _make_utf16(temp_dir, 'LumaViewPro-4.0.0-beta23_1_LVP.log', body)

    capture_installer_logs(log_dir, temp_dir=temp_dir)

    target = log_dir / 'install' / 'LumaViewPro-4.0.0-beta23_1_LVP.log'
    raw = target.read_bytes()
    assert raw == body.encode('utf-8'), 'content must survive the transcode intact'
    assert raw[:2] not in (b'\xff\xfe', b'\xfe\xff'), 'the byte-order mark must be dropped'
    assert target.stat().st_size < source.stat().st_size


def test_a_transcoded_log_is_not_recopied_every_startup(tmp_path):
    """The convergence check cannot compare sizes once a file is transcoded --
    the target is permanently smaller than its source, so a size comparison
    would never match and every startup would recopy every log forever."""
    temp_dir = tmp_path / 'temp'
    log_dir = tmp_path / 'logs'
    _make_utf16(temp_dir, 'LumaViewPro-4.0.0-beta23_2_LVP.log', 'body\r\n')

    first = capture_installer_logs(log_dir, temp_dir=temp_dir)
    second = capture_installer_logs(log_dir, temp_dir=temp_dir)

    assert first == ['LumaViewPro-4.0.0-beta23_2_LVP.log']
    assert second == [], 'a transcoded log must not be recaptured on the next startup'


def test_a_grown_utf16_log_is_recaptured(tmp_path):
    temp_dir = tmp_path / 'temp'
    log_dir = tmp_path / 'logs'
    source = _make_utf16(temp_dir, 'LumaViewPro-4.0.0-beta23_3_LVP.log', 'partial\r\n')
    capture_installer_logs(log_dir, temp_dir=temp_dir)

    source.write_bytes(b'\xff\xfe' + 'partial\r\nand the rest\r\n'.encode('utf-16-le'))
    again = capture_installer_logs(log_dir, temp_dir=temp_dir)

    assert again == ['LumaViewPro-4.0.0-beta23_3_LVP.log']
    target = log_dir / 'install' / 'LumaViewPro-4.0.0-beta23_3_LVP.log'
    assert target.read_bytes() == b'partial\r\nand the rest\r\n'


def test_a_non_utf16_log_is_copied_byte_for_byte(tmp_path):
    """The burn bundle log is already ASCII; it must not be rewritten."""
    temp_dir = tmp_path / 'temp'
    log_dir = tmp_path / 'logs'
    source = _make(temp_dir, 'LumaViewPro-4.0.0-beta23_4.log', content='plain ascii body')

    capture_installer_logs(log_dir, temp_dir=temp_dir)

    target = log_dir / 'install' / 'LumaViewPro-4.0.0-beta23_4.log'
    assert target.read_bytes() == source.read_bytes()


def test_a_bom_that_does_not_decode_is_copied_verbatim(tmp_path):
    """A log that cannot be transcoded is still evidence. Halving its size
    is never worth losing it, so the bytes go across untouched."""
    temp_dir = tmp_path / 'temp'
    log_dir = tmp_path / 'logs'
    # A BOM followed by an odd byte count cannot be valid UTF-16.
    corrupt = b'\xff\xfe' + b'a\x00b\x00\x41'
    directory = temp_dir
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'LumaViewPro-4.0.0-beta23_5_LVP.log').write_bytes(corrupt)

    copied = capture_installer_logs(log_dir, temp_dir=temp_dir)

    assert copied == ['LumaViewPro-4.0.0-beta23_5_LVP.log']
    target = log_dir / 'install' / 'LumaViewPro-4.0.0-beta23_5_LVP.log'
    assert target.read_bytes() == corrupt, 'an undecodable log must survive intact'
