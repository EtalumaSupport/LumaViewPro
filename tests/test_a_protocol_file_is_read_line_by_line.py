# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol file is read once, line by line, and takes what the whole-file read took.

The load read the whole file and copied its text five times before parsing
it; it now reads the file's lines as they come, through one source the
header block and the step table share. What the whole-file read did for
every line, that source does wherever the line falls: a line of whitespace
alone is skipped, and a byte that is not UTF-8 is refused as the file not
being text. (The shipped v2 example, with its blank line before 'Steps', is
`test_a_zero_step_file_loads_as_an_empty_protocol.py`'s stepped file.)
"""

import pytest

from _lumascope.modules.protocol import ProtocolFormatError
from tests.protocol_fixtures import protocol_from_file
from tests.test_protocol_roundtrip import _build_protocol, _make_step


def _saved(tmp_path):
    path = tmp_path / 'protocol.tsv'
    _build_protocol(
        [_make_step(name='A1_BF', well='A1'), _make_step(name='A2_BF', well='A2')]
    ).to_file(path)
    return path


def _lines(path):
    return path.read_bytes().split(b'\n')


def _with_line_before(path, starts, line):
    lines = _lines(path)
    at = next(i for i, text in enumerate(lines) if text.startswith(starts))
    path.write_bytes(b'\n'.join([*lines[:at], line, *lines[at:]]))


@pytest.mark.parametrize(
    'starts', [b'Period', b'A2_BF'], ids=['in-the-header-block', 'among-the-steps']
)
def test_a_line_of_whitespace_alone_is_skipped_wherever_it_falls(tmp_path, starts):
    path = _saved(tmp_path)
    _with_line_before(path, starts, b' \t ')

    protocol = protocol_from_file(file_path=path)

    assert protocol.steps()['Well'].tolist() == ['A1', 'A2']
    assert protocol.period_s() == 60


@pytest.mark.parametrize(
    'starts', [b'Labware', b'A2_BF'], ids=['in-the-header-block', 'among-the-steps']
)
def test_a_byte_that_is_not_utf8_is_refused_wherever_it_falls(tmp_path, starts):
    path = _saved(tmp_path)
    lines = _lines(path)
    at = next(i for i, text in enumerate(lines) if text.startswith(starts))
    lines[at] = lines[at].replace(b'\t', b'\xff\t', 1)
    path.write_bytes(b'\n'.join(lines))

    with pytest.raises(ProtocolFormatError, match='it is not a text file') as refused:
        protocol_from_file(file_path=path)

    assert refused.value.file == path


def test_a_file_with_windows_line_endings_loads(tmp_path):
    path = _saved(tmp_path)
    path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))

    protocol = protocol_from_file(file_path=path)

    assert protocol.steps()['Well'].tolist() == ['A1', 'A2']
    assert protocol.labware_id() == '6 well microplate'
