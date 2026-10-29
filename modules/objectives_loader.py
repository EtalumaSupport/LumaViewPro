# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import logging
import pathlib
import types
from collections.abc import Mapping
from dataclasses import dataclass

from modules.exceptions import (
    CatalogueNameRefusedError,
    InstallationFileError,
    ObjectiveUnknownError,
)
from modules.path_utils import read_installation_file, resolve_data_file
from modules.api_surface import api_fields

logger = logging.getLogger('LVP.modules.objectives_loader')

# The objective the objective question proposes when nothing names the glass
# in the light path: a fresh install, or a turret slot with no assignment. A
# proposal only -- the person answering confirms or changes it.
DEFAULT_PROPOSED_OBJECTIVE_ID = '20x w/collar'


@api_fields(
    'id',
    'description',
    'short_name',
    'magnification',
    'focal_length_mm',
    'numerical_aperture',
    'depth_of_field_um',
    'working_distance_mm',
    'autofocus_step_min_um',
    'autofocus_step_max_um',
    'autofocus_half_range_um',
    'jog_z_fine_um',
    'jog_z_coarse_um',
    'jog_xy_fine_um',
    'jog_xy_coarse_um',
    'z_settle_tolerance_um',
    'z_settle_timeout_ms',
    'focus_peak_fwhm_dense_um',
)
@dataclass(frozen=True)
class Objective:
    """One objective of the installation's catalogue, read once at bring-up.

    ``id`` is the catalogue key, the objective's one identity everywhere in
    the program; ``short_name`` the token step and file names carry, derived
    from the id alone (``objective_short_name``). Autofocus starts at
    ``autofocus_step_max_um`` and refines down to ``autofocus_step_min_um``
    across ``autofocus_half_range_um`` either side of where it starts. A
    jog moves the ``jog_*`` distance on its axis. ``focus_peak_fwhm_dense_um``
    is the objective's measured focus-peak width (full width at half
    maximum), None until it has been measured.
    """

    id: str
    description: str
    short_name: str
    magnification: float
    focal_length_mm: float
    numerical_aperture: float
    depth_of_field_um: float
    working_distance_mm: float
    autofocus_step_min_um: float
    autofocus_step_max_um: float
    autofocus_half_range_um: float
    jog_z_fine_um: float
    jog_z_coarse_um: float
    jog_xy_fine_um: float
    jog_xy_coarse_um: float
    z_settle_tolerance_um: float
    z_settle_timeout_ms: float
    focus_peak_fwhm_dense_um: float | None = None


# Each field's key in objectives.json. The file keeps the names it has
# always had, written by the engineering plugin's merge as well as by hand.
_FILE_KEYS = {
    'description': 'description',
    'magnification': 'magnification',
    'focal_length_mm': 'focal_length',
    'numerical_aperture': 'aperture',
    'depth_of_field_um': 'DOF',
    'working_distance_mm': 'working_distance',
    'autofocus_step_min_um': 'AF_min',
    'autofocus_step_max_um': 'AF_max',
    'autofocus_half_range_um': 'AF_range',
    'jog_z_fine_um': 'z_fine',
    'jog_z_coarse_um': 'z_coarse',
    'jog_xy_fine_um': 'xy_fine',
    'jog_xy_coarse_um': 'xy_coarse',
    'z_settle_tolerance_um': 'z_settle_tolerance_um',
    'z_settle_timeout_ms': 'z_settle_timeout_ms',
    'focus_peak_fwhm_dense_um': 'focus_peak_fwhm_dense_um',
}
_OPTIONAL_FILE_KEYS = frozenset({'focus_peak_fwhm_dense_um'})


def under_field_names(stated: Mapping) -> dict:
    """An objective as an image states it, each catalogue file key under its field's name.

    An image written before its objective was saved under the field names
    states it under the catalogue file's keys (``working_distance``,
    ``aperture``, ``DOF``, ...); one written since states the field names,
    which pass through, as does any key that is neither.
    """
    field_of = {key: field for field, key in _FILE_KEYS.items()}
    return {field_of.get(key, key): value for key, value in stated.items()}


def _read_objective(objective_id: object, entry: object, filepath: pathlib.Path) -> Objective:
    """The record for one catalogue entry, or a refusal naming the file, the objective and the key.

    Every declared key is required but the plugin's measured one, and a key
    no field declares is refused too: a misspelt key would otherwise leave
    its field at a value nobody wrote.
    """
    if not isinstance(objective_id, str) or not isinstance(entry, dict):
        raise InstallationFileError(
            filepath, f'has an entry {objective_id!r} that is not an objective'
        )
    undeclared = sorted(set(entry) - set(_FILE_KEYS.values()))
    if undeclared:
        raise InstallationFileError(
            filepath, f'gives objective {objective_id!r} the undeclared key {undeclared[0]!r}'
        )
    values = {}
    for field, key in _FILE_KEYS.items():
        if key not in entry:
            if key in _OPTIONAL_FILE_KEYS:
                continue
            raise InstallationFileError(filepath, f'gives objective {objective_id!r} no {key!r}')
        value = entry[key]
        if field == 'description':
            if not isinstance(value, str):
                raise InstallationFileError(
                    filepath, f'gives objective {objective_id!r} a {key!r} that is not text'
                )
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InstallationFileError(
                filepath, f'gives objective {objective_id!r} a {key!r} that is not a number'
            )
        values[field] = value
    return Objective(id=objective_id, short_name=objective_short_name(objective_id), **values)


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
        entries = read_installation_file(filepath)
        if not isinstance(entries, dict):
            raise InstallationFileError(filepath, 'is not a table of objectives')
        objectives = {
            objective_id: _read_objective(objective_id, entry, filepath)
            for objective_id, entry in entries.items()
        }
        if DEFAULT_PROPOSED_OBJECTIVE_ID not in objectives:
            raise InstallationFileError(
                filepath,
                f'has no {DEFAULT_PROPOSED_OBJECTIVE_ID!r}, the objective the objective '
                'question proposes by default',
            )
        # Two objectives with one short name would write their files under one name.
        owners: dict[str, str] = {}
        for objective in objectives.values():
            other = owners.setdefault(objective.short_name, objective.id)
            if other != objective.id:
                raise InstallationFileError(
                    filepath,
                    f'names {other!r} and {objective.id!r} with one short name '
                    f'{objective.short_name!r}',
                )
        self._objectives = types.MappingProxyType(objectives)

    @property
    def objectives(self) -> Mapping[str, Objective]:
        """The catalogue, by id; read-only, as each record is."""
        return self._objectives

    def get_objective_info(self, objective_id: str | None) -> Objective:
        """The catalogue entry for one objective, or a refusal naming why not.

        The catalogue key is the objective's one identity everywhere inside
        the program: the settings store, the turret slots, a protocol step
        and the spinner all carry it. The short name is a filename token
        derived from it and the magnification a button label; neither names
        an objective here.

        Raises:
            ObjectiveUnknownError: ``'none_selected'``, a null identifier.
            CatalogueNameRefusedError: ``'objective_not_in_catalogue'``, a
                non-string or a key the catalogue does not hold; ``offered``
                carries the keys it has.

        Both are ``ConfigError``s: the launch path recovers from exactly
        that type by republishing the shipped template, and an untyped raise
        escapes that recovery and takes app start down with it.
        """
        if objective_id is None:
            # A stored `objective_id` of null is a legal value on disk: the
            # settings shape gate passes null through deliberately, so this
            # has to be the settings failure it actually is.
            raise ObjectiveUnknownError('none_selected')

        # Exact key only. A prefix match used to stand in for a near miss, and
        # with '10x Oly' and '10x Phase' both in the catalogue an id of '10x'
        # bound silently to whichever came first in the file -- a real
        # objective with a real focal length answering for a name that fits
        # two. A near miss is refused by name so the file naming it gets fixed.
        if not isinstance(objective_id, str) or objective_id not in self._objectives:
            raise CatalogueNameRefusedError(
                'objective_not_in_catalogue',
                argument='objective_id',
                value=objective_id,
                offered=tuple(self._objectives),
            )

        return self._objectives[objective_id]

    def get_objectives_list(self) -> list:
        return list(self._objectives.keys())
