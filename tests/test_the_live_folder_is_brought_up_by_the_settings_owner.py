"""The live folder is made absolute and created by the settings owner, for every host.

The shipped template holds a relative live folder. Only the GUI's settings
panel used to resolve it, against the data directory, so a REST or script
host kept the relative value and its writers resolved it against whatever
working directory the process had; and the panel replaced a folder it could
not create with a default one, then saved the replacement over the person's
folder. The settings owner now brings the folder up on both of its paths --
the ordinary preparation and the late fall-back to the template -- and a
folder it cannot create stays the person's: captures into it are refused by
the location owner, naming it.
"""

import json
import logging
import os

import pytest

from modules import settings_init

_REPO_TEMPLATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'settings.json'
)


def _data_dir(tmp_path, live_folder=None):
    """A data directory holding the shipped template and, if given, a current.json."""
    data = tmp_path / 'data'
    data.mkdir()
    with open(_REPO_TEMPLATE) as f:
        template = json.load(f)
    with open(data / 'settings.json', 'w') as f:
        json.dump(template, f)
    if live_folder is not None:
        with open(data / 'current.json', 'w') as f:
            json.dump({**template, 'live_folder': live_folder}, f)
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_module_state():
    saved_settings = settings_init.settings
    saved_flag = settings_init.rejected_current_json
    yield
    settings_init.settings = saved_settings
    settings_init.rejected_current_json = saved_flag


def test_the_template_folder_is_made_absolute_under_the_data_directory_and_created(tmp_path):
    directory = _data_dir(tmp_path)

    settings, _ = settings_init.prepare_settings(
        logging.getLogger('t'), str(directory), fall_back_to_template=False
    )

    expected = (directory / 'capture').resolve()
    assert settings['live_folder'] == str(expected)
    assert expected.is_dir()


def test_an_absolute_folder_is_kept_as_written(tmp_path):
    chosen = tmp_path / 'elsewhere' / 'captures'
    directory = _data_dir(tmp_path, live_folder=str(chosen))

    settings, _ = settings_init.prepare_settings(
        logging.getLogger('t'), str(directory), fall_back_to_template=False
    )

    assert settings['live_folder'] == str(chosen)
    assert chosen.is_dir()


def test_a_folder_that_cannot_be_created_stays_the_persons(tmp_path, caplog):
    blocker = tmp_path / 'a_file'
    blocker.write_text('')
    unreachable = blocker / 'capture'
    directory = _data_dir(tmp_path, live_folder=str(unreachable))

    with caplog.at_level(logging.WARNING):
        settings, _ = settings_init.prepare_settings(
            logging.getLogger('t'), str(directory), fall_back_to_template=False
        )

    assert settings['live_folder'] == str(unreachable)
    assert not unreachable.exists()
    assert [r for r in caplog.records if str(unreachable) in r.getMessage()]


def test_the_late_fall_back_to_the_template_brings_the_folder_up_too(tmp_path):
    directory = _data_dir(tmp_path, live_folder=str(tmp_path / 'the_users'))
    settings_init.load_lvp_settings(logging.getLogger('t'), str(directory))

    settings_init.fall_back_to_template(logging.getLogger('t'), str(directory), 'a test')

    expected = (directory / 'capture').resolve()
    assert settings_init.settings['live_folder'] == str(expected)
    assert expected.is_dir()
