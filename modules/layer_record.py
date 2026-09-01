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

import json
from dataclasses import dataclass

from lvp_logger import logger
from modules.path_utils import resolve_data_file


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


@dataclass(frozen=True)
class LayerIdentity:
    """The resolved per-unit identity snapshot.

    `source` records which rung answered: 'motorconfig' (the unit's own
    block -- authoritative and complete when present), 'scopes' (the
    model's rows -- what a pre-block unit of this model has), or
    'unresolved' (no block and no resolvable model; empty, and loud at
    the point of LED use rather than silently wrong here).
    """

    layers: tuple[LayerRecord, ...]
    filterset: str
    source: str

    def find(self, key_name: str) -> LayerRecord | None:
        """Return the layer whose stable key name matches, else None.

        The deserialisation boundary: a name arriving from disk or from
        an API caller is mapped to its record exactly once, here.
        """
        for layer in self.layers:
            if layer.key_name == key_name:
                return layer
        return None


UNRESOLVED = LayerIdentity(layers=(), filterset='', source='unresolved')

# The release catalogue is loaded once per process: it ships with the
# release (the file is version-paired), so nothing invalidates it at
# runtime. Tests point the resolver at fixture files explicitly and may
# replace this cache to exercise a different vocabulary.
_CATALOGUE_CACHE: tuple[str, ...] | None = None


def release_catalogue() -> tuple[str, ...]:
    """The release's layer vocabulary (stable key names, display order).

    The single source every vocabulary consumer derives from -- layer
    lists, protocol validation, metadata channel acceptance.
    """
    global _CATALOGUE_CACHE
    if _CATALOGUE_CACHE is None:
        catalogue = load_layer_catalogue(load_scopes_data())
        if not catalogue:
            # A failed load stays uncached: memoizing it would turn one
            # transient bad context (a data root that resolved wrongly
            # for a single call) into an empty vocabulary for the whole
            # process, long after the context recovered. The load itself
            # already said what went wrong.
            return catalogue
        _CATALOGUE_CACHE = catalogue
    return _CATALOGUE_CACHE


# Row fields the resolver requires. Extra keys are tolerated so a config
# authored by a newer wizard still resolves on this release; a row
# MISSING one of these is unusable and is skipped loudly instead.
_REQUIRED_ROW_FIELDS = ('key_name', 'display_name', 'led_channel', 'excitation_nm')


def load_scopes_data(data_file: str | None = None) -> dict:
    """Load the scopes data file, returning {} (loudly) when unreadable.

    `data_file` exists so tests and tools can point the whole identity
    machinery at a fixture file; production callers pass nothing.
    """
    path = data_file if data_file is not None else resolve_data_file('scopes.json')
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        logger.error(f'[LAYER_RECORD] scopes data unreadable at {path}: {e}')
        return {}


def load_layer_catalogue(scopes_data: dict) -> tuple[str, ...]:
    """The release's layer vocabulary, in display order.

    The catalogue is the single authored order: a layer's id IS its
    position here, and every identity row (model or per-unit block) must
    name a catalogued key to resolve. Deriving the order from the
    per-model row lists instead would make id assignment depend on which
    model happens to be listed first, so the order is stated once.
    """
    raw = scopes_data.get('LayerOrder')
    if not isinstance(raw, list) or not raw or not all(isinstance(k, str) for k in raw):
        logger.error(f'[LAYER_RECORD] LayerOrder missing, empty, or malformed: {raw!r}')
        return ()
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
    override_model: str | None = None,
    data_file: str | None = None,
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
    the empty `unresolved` snapshot.

    A block that is absent because the board's config could not be READ
    is not the same as a unit with no block: the failed read is logged
    as an error here (the one place both facts are in hand) and the
    model rung answers, so the scope stays usable while the failure
    stays visible.
    """
    scopes_data = load_scopes_data(data_file)
    catalogue = load_layer_catalogue(scopes_data)
    models = scopes_data.get('Models')
    if not isinstance(models, dict):
        logger.error(f'[LAYER_RECORD] scopes data has no Models section: {sorted(scopes_data)!r}')
        models = {}

    def _from_model(model: str, source: str) -> LayerIdentity | None:
        entry = models.get(model)
        if not isinstance(entry, dict):
            return None
        layers = _parse_rows(entry.get('Layers', []), catalogue, f'scopes[{model}]')
        filterset = entry.get('Filterset', '')
        if not isinstance(filterset, str):
            filterset = ''
        return LayerIdentity(layers=layers, filterset=filterset, source=source)

    if override_model is not None:
        logger.warning(
            f'[LAYER_RECORD] identity override active: resolving as model '
            f'{override_model!r} for this session'
        )
        identity = _from_model(override_model, source='scopes')
        if identity is not None:
            return identity
        logger.error(
            f'[LAYER_RECORD] override model {override_model!r} has no scopes '
            f'entry; identity is unresolved'
        )
        return UNRESOLVED

    if board_block is not None:
        layers = _parse_rows(board_block.get('Layers', []), catalogue, 'motorconfig')
        filterset = board_block.get('Filterset', '')
        if not isinstance(filterset, str):
            filterset = ''
        return LayerIdentity(layers=layers, filterset=filterset, source='motorconfig')

    if not board_config_read_ok:
        logger.error(
            '[LAYER_RECORD] board config could not be read; a per-unit LED '
            'block may exist but is unavailable -- resolving from the model '
            'instead'
        )

    # The configured model is consulted only when the hardware reports no
    # model at all. A motor-reported model with no scopes entry (a newer
    # unit than this release knows) goes unresolved and loud rather than
    # silently adopting whatever the user last selected.
    model = motor_model or configured_model
    if model:
        identity = _from_model(model, source='scopes')
        if identity is not None:
            return identity
        logger.error(f'[LAYER_RECORD] model {model!r} has no scopes entry; identity is unresolved')

    return UNRESOLVED
