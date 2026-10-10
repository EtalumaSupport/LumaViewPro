# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Layer identity: what a layer IS on the attached unit.

One record per layer (stable key name, display name, LED board address,
excitation wavelength) plus the unit's filterset, resolved once into an
immutable snapshot. Identity is unit data, not code: different filtersets
carry different LEDs, so the truth lives in the unit's motorconfig LED
block when it has one, and in the model's `scopes.json` rows for units
built before per-unit blocks existed.

Internally a layer is identified by its integer id. The names are fields
that get WRITTEN OUT (display, and serialisation under the stable
`key_name`); nothing in memory looks a layer up by its name past the
deserialisation boundary this module implements.

Resolution never raises: a scope with no resolvable identity gets the
empty `unresolved` snapshot, and the illumination API is where that
state becomes a loud, named error on first use. Failing construction
here would take down a scope whose camera and stage are fine.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from lvp_logger import logger
from modules.exceptions import (
    ArgumentRefusedError,
    ConfigError,
    HardwareCommandRefusedError,
    InstallationFileError,
    MissingPart,
)
from modules.path_utils import read_installation_file, resolve_data_file
from modules.api_surface import api, api_fields


@api_fields('display_name', 'excitation_nm', 'id', 'key_name', 'led_channel')
@dataclass(frozen=True)
class LayerRecord:
    """Identity of one layer on one unit.

    `id` is the internal key and the display position -- assigned from
    the release catalogue's order, never persisted to any per-unit or
    user file, so it can renumber freely between releases.

    `key_name` is the stable serialisation name (settings block keys,
    protocol TSV `Color`, filename tokens, metadata `channel`). It never
    changes when a layer is renamed.

    `display_name` is what the operator sees. Freely changeable: a
    rename touches this field and nothing persisted.

    `led_channel` is the LED board address(es) this layer drives, empty
    for a layer with no LED (luminescence). A tuple rather than a single
    value because a future channel (quantitative phase) drives several
    switches as one layer; today every shipped row carries zero or one.

    `excitation_nm` is the excitation wavelength -- named for what it
    holds, because the layer NAME is the emission colour the operator
    sees on screen, and an ambiguous "wavelength" on a layer called
    Green invites silently wrong metadata. None is the truth for
    broadband transmitted light and for LED-less layers.
    """

    id: int
    key_name: str
    display_name: str
    led_channel: tuple[int, ...]
    excitation_nm: float | None


@api_fields('filterset', 'layers', 'model', 'source')
@dataclass(frozen=True)
class LayerIdentity:
    """The resolved per-unit identity snapshot.

    `source` records which rung answered: 'motorconfig' (the unit's own
    block -- authoritative and complete when present), 'scopes' (the
    model's rows -- what a pre-block unit of this model has), or
    'unresolved' (no block and no resolvable model; empty, and loud at
    the point of LED use rather than silently wrong here).

    `model` is the scope model the resolver settled on, whichever rung
    answered: the override, else the model the board reports, else the
    operator's selection -- the only identity an FX2 scope has, since it
    cannot report its own. A model the catalogue lacks is still carried,
    with no layers; `None` means there was no model at all. Every reader
    of "which model is this scope" takes it from here, so the layers, the
    capabilities and the saved files name one model.
    """

    layers: tuple[LayerRecord, ...]
    filterset: str
    source: str
    model: str | None

    @api(in_process=True)
    def find(self, key_name: str) -> LayerRecord | None:
        """Return the layer whose stable key name matches, else None.

        The deserialisation boundary: a name arriving from disk or from
        an API caller is mapped to its record exactly once, here.
        """
        for layer in self.layers:
            if layer.key_name == key_name:
                return layer
        return None

    @api(in_process=True)
    def refuse_unless_on_scope(self, layer: str, member: str, *, then: str) -> None:
        """Refuse ``layer`` unless it is a layer of this release and this unit has it.

        The one question for a command that takes a layer: its settings,
        its focus, a still or a recording named for it. The name first
        (``refuse_unknown_layer``), so a name that is no layer is never
        told it is missing hardware.

        Raises:
            ArgumentRefusedError: ``'layer_unknown'``.
            HardwareCommandRefusedError: ``'axis_absent'``, naming the
                layer (``MissingPart.layer``).
            ConfigError: this unit's layers could not be resolved, so
                ``layer`` cannot ``then``.
        """
        refuse_unknown_layer(layer)
        if self.find(layer) is not None:
            return
        if self.layers:
            part = MissingPart.layer(layer)
            raise HardwareCommandRefusedError(part.reason, member, missing=part)
        raise ConfigError(
            f"this scope's layers could not be resolved (model {self.model}), "
            f'so {layer} cannot {then}'
        )


UNRESOLVED = LayerIdentity(layers=(), filterset='', source='unresolved', model=None)

# The release catalogue is loaded once per process: it ships with the
# release (the file is version-paired), so nothing invalidates it at
# runtime. Tests hand the resolver a vocabulary explicitly and may
# replace this cache to exercise a different one.
_CATALOGUE_CACHE: tuple[str, ...] | None = None


def refuse_unknown_layer(layer: object, argument: str = 'layer') -> None:
    """Refuse a value that is not one of this release's layers.

    The one check of a layer name, asked by every member that takes one
    before it asks whether the unit has the layer: a name that is no layer
    is the request's fault on every scope. Names are exact: ``'bf'`` is
    not ``'BF'``.

    Raises:
        ArgumentRefusedError: ``'no_layer_selected'`` for None,
            ``'layer_unknown'`` for anything else not in the catalogue;
            each offers the catalogue.
    """
    catalogue = release_catalogue()
    if layer is None:
        raise ArgumentRefusedError(
            'no_layer_selected', argument=argument, value=layer, offered=catalogue
        )
    if not isinstance(layer, str) or layer not in catalogue:
        raise ArgumentRefusedError(
            'layer_unknown', argument=argument, value=layer, offered=catalogue
        )


def release_catalogue() -> tuple[str, ...]:
    """The release's layer vocabulary (stable key names, display order).

    The single source every vocabulary consumer derives from -- layer
    lists, protocol validation, metadata channel acceptance. Read from
    the installation's own folder, not a scope's: the settings bring-up
    needs it before any scope exists.

    Raises:
        InstallationFileError: the installation's ``scopes.json`` is
            unusable or states no layer order. Nothing is cached, so the
            next call reads the file again.
    """
    global _CATALOGUE_CACHE
    if _CATALOGUE_CACHE is None:
        path = resolve_data_file('scopes.json')
        _CATALOGUE_CACHE = load_layer_catalogue(read_installation_file(path), path)
    return _CATALOGUE_CACHE


# Row fields the resolver requires. Extra keys are tolerated so a config
# authored by a newer wizard still resolves on this release; a row
# MISSING one of these is unusable and is skipped loudly instead.
_REQUIRED_ROW_FIELDS = ('key_name', 'display_name', 'led_channel', 'excitation_nm')


# The fields every model entry states, with their types. A missing or
# mistyped one is a warning, not a refusal: `entry_axes` reads a missing
# flag as absent and a mistyped one by its truth, so the warning is where
# such an entry shows. `LEDBoard` names the board that drives the model's
# LEDs; production reads it nowhere, since the bring-up finds the board it
# has, and only the simulator builds from it. `MotorBoard` is not listed:
# a manual model has no motor board to name.
_MODEL_ENTRY_FIELDS = {
    'Focus': bool,
    'XYStage': bool,
    'Turret': bool,
    'Layers': list,
    'LEDBoard': str,
}


def load_scope_models(data_file: str | None = None) -> Mapping:
    """The model catalogue -- scopes.json's `Models` section -- or a refusal.

    The scope reads it once, from the folder it was started on, before
    anything is started, and everything else reads the scope's copy. With
    no catalogue the declared model has no entry, `model_has_turret`
    answers False, and a turret scope whose board is not talking is
    configured as turretless -- answering with its stored objective
    instead of the one in the light path. So an unusable file or a
    missing section refuses, naming the file.

    ``data_file`` None reads the installation's own folder.

    Raises:
        InstallationFileError: the file is unusable, has no Models section,
            or has a model entry that is not an object.
    """
    path = data_file if data_file is not None else resolve_data_file('scopes.json')
    models = read_installation_file(path).get('Models')
    if not isinstance(models, dict):
        raise InstallationFileError(
            path, f'has no usable Models section (found {type(models).__name__})'
        )
    for model, entry in models.items():
        # Every reader of an entry asks it for its fields, so one that is
        # not an object would fail wherever it was first read.
        if not isinstance(entry, dict):
            raise InstallationFileError(
                path,
                f'has a model {model!r} whose entry is a {type(entry).__name__}, not an object',
            )
        for field, expected_type in _MODEL_ENTRY_FIELDS.items():
            if field not in entry:
                logger.warning(f"[LAYER_RECORD] model '{model}' missing '{field}' in {path}")
            elif not isinstance(entry[field], expected_type):
                logger.warning(
                    f"[LAYER_RECORD] model '{model}'.'{field}' should be "
                    f'{expected_type.__name__}, got {type(entry[field]).__name__} in {path}'
                )
    return models


# The catalogue's three hardware flags and the motor axes each one stands
# for: a focus drive is Z, a stage is X and Y, a turret is T.
_AXES_BY_FLAG = (('Focus', 'Z'), ('XYStage', 'XY'), ('Turret', 'T'))


def model_axes(models: dict, model: str) -> frozenset[str]:
    """The motor axes the catalogue says a model has, or a refusal.

    A model the catalogue does not list is refused rather than answered
    as axis-less: an axis-less answer builds a manual scope, and a typo in
    the model name would then look like a scope with no motor board.
    """
    entry = models.get(model)
    if not isinstance(entry, dict):
        raise ConfigError(
            f'scopes.json lists no model {model!r} (known: {sorted(models)}); '
            'the microscope setting names a model the catalogue lacks'
        )
    return entry_axes(entry)


def entry_axes(entry: dict) -> frozenset[str]:
    """The motor axes one catalogue entry declares. Empty for a manual
    scope, which has no motor board at all."""
    return frozenset(axis for flag, axes in _AXES_BY_FLAG if entry.get(flag) for axis in axes)


def entry_expects_motion(entry: dict | None) -> bool:
    """Whether a scope with this catalogue entry has a motor board.

    No entry expects one: only the catalogue can say a scope is manual,
    and answering "manual" for a model it does not list would let a
    motorized scope's missing board pass as expected.
    """
    return entry is None or bool(entry_axes(entry))


def load_layer_catalogue(scopes_data: dict, path: str | os.PathLike) -> tuple[str, ...]:
    """The release's layer vocabulary, in display order, or a refusal naming ``path``.

    The catalogue is the single authored order: a layer's id IS its
    position here, and every identity row (model or per-unit block) must
    name a catalogued key to resolve. Deriving the order from the
    per-model row lists instead would make id assignment depend on which
    model happens to be listed first, so the order is stated once. An
    empty vocabulary would let the settings check pass with no layers to
    check, so a missing one refuses.

    Raises:
        InstallationFileError: ``LayerOrder`` is missing, empty, or not a
            list of names.
    """
    raw = scopes_data.get('LayerOrder')
    if not isinstance(raw, list) or not raw or not all(isinstance(k, str) for k in raw):
        raise InstallationFileError(path, f'has no usable LayerOrder (found {raw!r})')
    return tuple(raw)


def _parse_rows(rows: object, catalogue: tuple[str, ...], origin: str) -> tuple[LayerRecord, ...]:
    """Parse identity rows, skipping each unusable row loudly.

    A bad row costs that row, never the scope: the surviving layers keep
    working and the skipped one is absent from identity (so its LED use
    is a loud, named error downstream). Falling back to another data
    source instead would silently describe hardware the unit does not
    have. Skipped-loudly cases: a row shape this release cannot read
    (the same forward-looking posture that lets an old release ignore a
    newer block), a key name outside the catalogue (an OEM/custom layer
    this release has no seat for), and a multi-address `led_channel`
    (representable, but drive semantics for it are not built yet).
    """
    if not isinstance(rows, list):
        logger.error(f'[LAYER_RECORD] {origin}: Layers is not a list: {rows!r}')
        return ()
    records = []
    for row in rows:
        if not isinstance(row, dict):
            logger.error(f'[LAYER_RECORD] {origin}: row is not a mapping: {row!r}')
            continue
        missing = [k for k in _REQUIRED_ROW_FIELDS if k not in row]
        if missing:
            logger.error(f'[LAYER_RECORD] {origin}: row {row!r} missing {missing}; skipped')
            continue
        key_name = row['key_name']
        if key_name not in catalogue:
            logger.error(
                f'[LAYER_RECORD] {origin}: layer {key_name!r} is not in this '
                f'release catalogue {catalogue}; skipped'
            )
            continue
        raw_channel = row['led_channel']
        if raw_channel is None:
            channel: tuple[int, ...] = ()
        elif isinstance(raw_channel, int) and not isinstance(raw_channel, bool):
            channel = (raw_channel,)
        elif isinstance(raw_channel, list) and all(
            isinstance(c, int) and not isinstance(c, bool) for c in raw_channel
        ):
            channel = tuple(raw_channel)
        else:
            logger.error(
                f'[LAYER_RECORD] {origin}: layer {key_name!r} has malformed '
                f'led_channel {raw_channel!r}; skipped'
            )
            continue
        if len(channel) > 1:
            logger.error(
                f'[LAYER_RECORD] {origin}: layer {key_name!r} drives multiple '
                f'channels {channel}; multi-channel layers are not supported '
                f'yet; skipped'
            )
            continue
        raw_nm = row['excitation_nm']
        if raw_nm is None:
            excitation: float | None = None
        elif isinstance(raw_nm, (int, float)) and not isinstance(raw_nm, bool):
            excitation = float(raw_nm)
        else:
            logger.error(
                f'[LAYER_RECORD] {origin}: layer {key_name!r} has malformed '
                f'excitation_nm {raw_nm!r}; skipped'
            )
            continue
        display = row['display_name']
        if not isinstance(display, str) or not display:
            logger.error(
                f'[LAYER_RECORD] {origin}: layer {key_name!r} has malformed '
                f'display_name {display!r}; skipped'
            )
            continue
        records.append(
            LayerRecord(
                id=catalogue.index(key_name),
                key_name=key_name,
                display_name=display,
                led_channel=channel,
                excitation_nm=excitation,
            )
        )
    return tuple(sorted(records, key=lambda r: r.id))


def resolve_layer_identity(
    *,
    board_block: dict | None,
    board_config_read_ok: bool,
    motor_model: str | None,
    configured_model: str | None,
    models: Mapping,
    catalogue: tuple[str, ...],
    override_model: str | None = None,
) -> LayerIdentity:
    """Resolve the unit's layer identity from the first authoritative source.

    Precedence: an explicit `override_model` (a lab/engineering request
    to impersonate a model for this session -- it wins over everything,
    loudly, and is never persisted); else the unit's own motorconfig LED
    block, taken WHOLE (a block describes one physical filterset
    assembly, so merging it field-by-field with model data could
    describe hardware that does not exist); else the model's
    `scopes.json` rows, with the motor-reported model outranking the
    configured one because hardware truth beats a user selection; else
    the empty `unresolved` snapshot. Whichever rung answers, the snapshot
    carries the model the resolver settled on (`LayerIdentity.model`).

    A block that is absent because the board's config could not be READ
    is not the same as a unit with no block: the failed read is logged
    as an error here (the one place both facts are in hand) and the
    model rung answers, so the scope stays usable while the failure
    stays visible.

    Reads nothing: ``models`` is the scope's model catalogue and
    ``catalogue`` the release's layer vocabulary, each read once by its
    owner, so rows are resolved against the one vocabulary every layer
    list in the process uses.
    """

    def _from_model(model: str) -> LayerIdentity:
        entry = models.get(model)
        if not isinstance(entry, dict):
            return LayerIdentity(layers=(), filterset='', source='unresolved', model=model)
        layers = _parse_rows(entry.get('Layers', []), catalogue, f'scopes[{model}]')
        filterset = entry.get('Filterset', '')
        if not isinstance(filterset, str):
            filterset = ''
        return LayerIdentity(layers=layers, filterset=filterset, source='scopes', model=model)

    if override_model is not None:
        logger.warning(
            f'[LAYER_RECORD] identity override active: resolving as model '
            f'{override_model!r} for this session'
        )
        identity = _from_model(override_model)
        if identity.source == 'unresolved':
            logger.error(
                f'[LAYER_RECORD] override model {override_model!r} has no scopes '
                f'entry; identity is unresolved'
            )
        return identity

    # The configured model is consulted only when the hardware reports no
    # model at all. A motor-reported model with no scopes entry (a newer
    # unit than this release knows) goes unresolved and loud rather than
    # silently adopting whatever the user last selected.
    model = motor_model or configured_model

    if board_block is not None:
        layers = _parse_rows(board_block.get('Layers', []), catalogue, 'motorconfig')
        filterset = board_block.get('Filterset', '')
        if not isinstance(filterset, str):
            filterset = ''
        return LayerIdentity(layers=layers, filterset=filterset, source='motorconfig', model=model)

    if not board_config_read_ok:
        logger.error(
            '[LAYER_RECORD] board config could not be read; a per-unit LED '
            'block may exist but is unavailable -- resolving from the model '
            'instead'
        )

    if not model:
        return UNRESOLVED
    identity = _from_model(model)
    if identity.source == 'unresolved':
        logger.error(f'[LAYER_RECORD] model {model!r} has no scopes entry; identity is unresolved')
    return identity
