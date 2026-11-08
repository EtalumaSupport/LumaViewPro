# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every file the installation ships fails the same way: one error naming the file.

The labware and objective catalogues, the scope model catalogue and the
motor defaults are part of the installation, not the user's settings. Each
used to map a bad file to its own error -- `InstallationFileError`,
`ConfigError`, `RuntimeError` or an empty dict -- so a host could not tell
a broken installation from bad settings, and the GUI answered one of them by
replacing the user's settings with the template. One reader now opens them
all, and every way the file can be unusable is the one error.
"""

import logging

import pytest

from _lumascope.modules.exceptions import ConfigError, InstallationFileError
from _lumascope.modules.path_utils import read_installation_file

CASES = [
    ('missing', None),
    ('not-json', '{not json'),
    ('a-list', '[1, 2]'),
    ('a-string', '"text"'),
    ('not-utf8', b'{"a": "\xff"}'),
]


@pytest.mark.parametrize(('case', 'content'), CASES, ids=[c for c, _ in CASES])
def test_an_unusable_file_is_one_error_naming_it(tmp_path, caplog, case, content):
    path = tmp_path / 'data' / 'scopes.json'
    path.parent.mkdir()
    if isinstance(content, bytes):
        path.write_bytes(content)
    elif content is not None:
        path.write_text(content, encoding='utf-8')

    with caplog.at_level(logging.DEBUG), pytest.raises(InstallationFileError) as failure:
        read_installation_file(path)

    assert failure.value.file_path == path
    assert 'scopes.json' in str(failure.value)
    assert str(path.parent) in str(failure.value)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], (
        'the reader logged the failure itself; whoever catches it logs it once'
    )


def test_a_path_that_cannot_be_opened_is_the_same_error(tmp_path):
    path = tmp_path / 'motorconfig_defaults.json'
    path.mkdir()

    with pytest.raises(InstallationFileError, match='cannot be read') as failure:
        read_installation_file(path)

    assert failure.value.file_path == path


def test_a_usable_file_is_its_object(tmp_path):
    path = tmp_path / 'labware.json'
    path.write_text('{"Wellplate": {}}', encoding='utf-8')

    assert read_installation_file(path) == {'Wellplate': {}}


def test_the_error_is_not_a_settings_error():
    assert not issubclass(InstallationFileError, ConfigError)


def test_json_no_installation_file_can_mean_is_refused_naming_what(tmp_path):
    """json.load keeps the last of two equal keys and reads NaN, Infinity and
    1e400 as floats, all without a word: a pasted plate that kept its name
    replaced the shipped one. The reader refuses each, naming the file and
    what it holds; an integer or nesting the parser cannot hold is the same
    error, not a ValueError or RecursionError from inside the parser."""
    refused = {
        'a key twice': ('{"Wellplate": {}, "Wellplate": {}}', "the key 'Wellplate' twice"),
        'a key twice, nested': ('{"a": {"x": 1, "x": 2}}', "the key 'x' twice"),
        'a key twice, in a list': ('{"a": [{"x": 1, "x": 2}]}', "the key 'x' twice"),
        'NaN': ('{"a": NaN}', 'NaN, which is not a number'),
        'Infinity': ('{"a": -Infinity}', '-Infinity, which is not a number'),
        'a number past a float': ('{"a": 1e400}', '1e400, which is not finite'),
        'an integer past the digit limit': ('{"a": ' + '9' * 5000 + '}', 'could not be parsed'),
        'nesting past the recursion limit': (
            '{"a": ' + '[' * 100_000 + ']' * 100_000 + '}',
            'could not be parsed',
        ),
    }
    path = tmp_path / 'labware.json'
    wrong = {}
    for case, (text, why) in refused.items():
        path.write_text(text, encoding='utf-8')
        try:
            read_installation_file(path)
            wrong[case] = 'accepted'
        except InstallationFileError as e:
            if e.file_path != path or why not in str(e):
                wrong[case] = str(e)
        except Exception as e:  # the reader let a parser error escape unnamed
            wrong[case] = f'{type(e).__name__}: {e}'[:200]

    assert wrong == {}
