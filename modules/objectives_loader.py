# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import json
import logging
import pathlib

import pandas as pd

from modules.exceptions import ConfigError
from modules.path_utils import resolve_data_file

logger = logging.getLogger('LVP.modules.objectives_loader')


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


def _validate_objectives(objectives: dict, filepath: str) -> None:
    """Validate objectives.json structure: each entry must have required fields."""
    if not isinstance(objectives, dict):
        raise ValueError(
            f'objectives.json at {filepath}: expected dict, got {type(objectives).__name__}'
        )
    for obj_id, obj in objectives.items():
        if not isinstance(obj, dict):
            raise ValueError(f"objectives.json: objective '{obj_id}' must be a dict")
        for field, expected_type in _REQUIRED_OBJECTIVE_FIELDS.items():
            if field not in obj:
                logger.warning(f"[Objectives] '{obj_id}' missing field '{field}' in {filepath}")
            elif not isinstance(obj[field], expected_type):
                logger.warning(
                    f"[Objectives] '{obj_id}'.'{field}' should be "
                    f'{expected_type}, got {type(obj[field]).__name__} in {filepath}'
                )


class ObjectiveLoader:
    def __init__(self, *arg, source_path: str | pathlib.Path | None = None):
        filepath = resolve_data_file('objectives.json', source_path=source_path)
        try:
            with open(filepath) as read_file:
                self._objectives = json.load(read_file)
        except FileNotFoundError as e:
            logger.error(f'[Objectives] objectives.json not found at {filepath}')
            raise RuntimeError(
                f'Required file objectives.json not found at {filepath}. '
                'Please reinstall or restore from backup.'
            ) from e
        except json.JSONDecodeError as e:
            logger.error(f'[Objectives] objectives.json is corrupt: {e}')
            raise RuntimeError(
                f'objectives.json is corrupt ({e}). Please restore from backup or reinstall.'
            ) from e

        _validate_objectives(self._objectives, filepath)
        self._generate_short_names()
        self._objectives_df = pd.DataFrame.from_dict(self._objectives, orient='index')

    def _create_short_name_from_objective_id(self, objective_id: str) -> str:

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

    def _generate_short_names(self):
        # Generate short name to be used for protocol step names
        for objective_key, objective_info in self._objectives.items():
            if 'short_name' not in objective_info:
                short_name = self._create_short_name_from_objective_id(objective_id=objective_key)
                self._objectives[objective_key]['short_name'] = short_name

        # Confirm there are no collisions
        short_names = [v['short_name'] for v in self._objectives.values()]
        short_names_set = set(short_names)
        if len(short_names_set) < len(short_names):
            raise Exception('Duplicate short names for objectives were generated')

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

    def get_objectives_list(self) -> list:
        return list(self._objectives.keys())

    def get_objectives_dataframe(self) -> pd.DataFrame:
        return self._objectives_df
