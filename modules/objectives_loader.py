# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import logging
import pathlib

import pandas as pd

from modules.exceptions import ConfigError, InstallationFileError
from modules.path_utils import read_installation_file, resolve_data_file
from modules.api_surface import api

logger = logging.getLogger('LVP.modules.objectives_loader')

# The objective the objective question proposes when nothing names the glass
# in the light path: a fresh install, or a turret slot with no assignment. A
# proposal only -- the person answering confirms or changes it.
DEFAULT_PROPOSED_OBJECTIVE_ID = '20x w/collar'


_REQUIRED_OBJECTIVE_FIELDS = {
    'description': str,
    'magnification': (int, float),
    'focal_length': (int, float),
    'aperture': (int, float),
    'DOF': (int, float),
    'working_distance': (int, float),
    'AF_min': (int, float),
    'AF_max': (int, float),
    'AF_range': (int, float),
    'z_fine': (int, float),
    'z_coarse': (int, float),
    'xy_fine': (int, float),
    'xy_coarse': (int, float),
}


def _validate_objectives(objectives: dict, filepath: pathlib.Path) -> None:
    """Refuse a catalogue whose shape no lookup can use; warn on an entry missing a field."""
    for obj_id, obj in objectives.items():
        if not isinstance(obj, dict):
            raise InstallationFileError(
                filepath, f'has an entry {obj_id!r} that is not an objective'
            )
        for field, expected_type in _REQUIRED_OBJECTIVE_FIELDS.items():
            if field not in obj:
                logger.warning(f"[Objectives] '{obj_id}' missing field '{field}' in {filepath}")
            elif not isinstance(obj[field], expected_type):
                logger.warning(
                    f"[Objectives] '{obj_id}'.'{field}' should be "
                    f'{expected_type}, got {type(obj[field]).__name__} in {filepath}'
                )


def objective_short_name(objective_id: str) -> str:
    """The objective's token in step names and file names, derived from its id alone.

    One function for every writer of a name -- the catalogue at capture and
    post-processing afterwards -- so one objective is never named two ways,
    and a run's files can be named from the id they recorded without any
    catalogue, on any installation.

    Raises:
        TypeError: ``objective_id`` is not a string; a missing id is the
            caller's to handle, never coerced into a name.
    """
    if not isinstance(objective_id, str):
        raise TypeError(f'an objective id is a string, got {type(objective_id).__name__}')

    tmp = objective_id.replace('w/o', 'No')
    tmp = tmp.replace('W/o', 'No')
    tmp = tmp.replace('W/O', 'No')

    tmp = tmp.replace('w/', '')
    tmp = tmp.replace('W/', '')

    # Remove illegal path characters
    tmp = tmp.replace('/', '')
    tmp = tmp.replace('\\', '')
    tmp = tmp.replace('-', '')
    tmp = tmp.replace('_', '')

    # Split on whitespace
    tmp = tmp.split(' ')

    # Capitalize the first letter of each word
    tmp = [v.capitalize() for v in tmp]

    # Rejoin into single key
    tmp = ''.join(tmp)

    return tmp


class ObjectiveLoader:
    def __init__(self, *arg, source_path: str | pathlib.Path | None = None):
        filepath = resolve_data_file('objectives.json', source_path=source_path)
        self._objectives = read_installation_file(filepath)

        _validate_objectives(self._objectives, filepath)
        if DEFAULT_PROPOSED_OBJECTIVE_ID not in self._objectives:
            raise InstallationFileError(
                filepath,
                f'has no {DEFAULT_PROPOSED_OBJECTIVE_ID!r}, the objective the objective '
                'question proposes by default',
            )
        self._generate_short_names(filepath)
        self._objectives_df = pd.DataFrame.from_dict(self._objectives, orient='index')

    def _generate_short_names(self, filepath: pathlib.Path):
        # Always derived, never read from the file: post-processing derives
        # the same token from a run's recorded id without a catalogue, so a
        # name written here must be the one it would derive.
        for objective_key, objective_info in self._objectives.items():
            objective_info['short_name'] = objective_short_name(objective_key)

        # Two objectives with one short name would write their files under one name.
        owners: dict[str, str] = {}
        for objective_key, objective_info in self._objectives.items():
            other = owners.setdefault(objective_info['short_name'], objective_key)
            if other != objective_key:
                raise InstallationFileError(
                    filepath,
                    f'names {other!r} and {objective_key!r} with one short name '
                    f'{objective_info["short_name"]!r}',
                )

    def get_objective_info(self, objective_id: str | None) -> dict:
        """The catalogue entry for one objective, or a refusal naming why not.

        The catalogue key is the objective's one identity everywhere inside
        the program: the settings store, the turret slots, a protocol step
        and the spinner all carry it. The short name is a filename token
        derived from it and the magnification a button label; neither names
        an objective here.

        Raises:
            ConfigError: A null identifier, a non-string, or a key the
                catalogue does not hold. One type for every unusable id: the
                launch path recovers from exactly this type by republishing
                the shipped template, and an untyped raise escapes that
                recovery and takes app start down with it.
        """
        if objective_id is None:
            # A stored `objective_id` of null is a legal value on disk: the
            # settings shape gate passes null through deliberately, so this
            # has to be the settings failure it actually is.
            raise ConfigError('no objective identifier supplied')

        # Exact key only. A prefix match used to stand in for a near miss, and
        # with '10x Oly' and '10x Phase' both in the catalogue an id of '10x'
        # bound silently to whichever came first in the file -- a real
        # objective with a real focal length answering for a name that fits
        # two. A near miss is refused by name so the file naming it gets fixed.
        if not isinstance(objective_id, str) or objective_id not in self._objectives:
            raise ConfigError(f'unknown objective {objective_id!r}; the catalogue has no such key')

        return self._objectives[objective_id]

    @api
    def get_objectives_list(self) -> list:
        return list(self._objectives.keys())

    @api
    def get_objectives_dataframe(self) -> pd.DataFrame:
        """The objectives table, as the caller's own copy.

        A changed table would otherwise change the catalogue every later
        reader gets.
        """
        return self._objectives_df.copy(deep=True)
