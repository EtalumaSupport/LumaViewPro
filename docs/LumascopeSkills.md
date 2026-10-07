# LumaViewPro — API & Integration Reference

**What is public.** The public surface is defined in code: a member is
public unless its docstring marks it internal ("not part of the L2 API
surface") or an engineering ruling places it on the bench/tech-support
surface. This document is the reference for that surface, and the test
suite holds the two in lockstep both ways -- every call form here must
resolve on the live API, and every public member must appear here. A
member absent from this document is internal or engineering surface:
it may be renamed, moved, or removed in any release without notice,
and code that calls it is unsupported.

## PRE-RELEASE API

The Lumascope SDK API documented in this file is **subject to breaking changes** in 4.1 / 4.1.5 / 4.2. Specifically:

- 4.1.5 ships the sub-API decomposition (Wave 7): hardware-direct methods on `Lumascope` move to sub-APIs (`scope.motion.*`, `scope.illumination.*`, `scope.imaging.*`, `scope.diagnostics.*`, `scope.capabilities.*`, `scope.io.*`). The `Lumascope` class becomes a thin facade; L2 entry point shifts to `ScopeSession`.
- 4.2 ships the capability + wire contract changes that may rename or restructure protocol-level surfaces.
- The REST endpoint convention is **deferred** to a dedicated design session; do not assume current shapes are final.

If you are using this API before stabilization, **contact Etaluma support** so we know to consult you before structural changes. Internal LumaViewPro use does not trigger this requirement.

The warning retires when the first non-`-beta` LumaViewPro release ships (the `4.0.0` git tag on the `4.0.0-beta` lineage). At that point the L2-callable surface freezes for the 4.x major version and every subsequent change is recorded in the [Changelog](#changelog) section below (additive / behavior-change / rename / removal, with version + justification). Until `4.0.0` ships, the API surface stays structurally fluid -- methods may be renamed, moved into sub-APIs, or retired without a deprecation cycle.

---

## Overview

LumaViewPro controls Etaluma microscopes: LED illumination, XYZ stage + turret motion, and camera image acquisition. This document is the integration reference for developers building scripts, headless automation, or external control applications on top of LumaViewPro.

**Repository**: `EtalumaSupport/LumaViewPro`
**Platform**: Python 3.12–3.13, Windows / macOS / Linux

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Your Application                               │
│  (MATLAB, Python script, LabVIEW, web app)      │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  REST surface  (HTTP/JSON, any language)        │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  ScopeSession session layer  (Python, headless) │
│  └─ executor-routed commands, protocol runner   │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  Lumascope composition root  (Python)           │
│  ├─ scope.motion        ├─ scope.capabilities   │
│  ├─ scope.illumination  ├─ scope.runtime_state  │
│  ├─ scope.imaging       ├─ scope.protocols      │
│  └─ scope.diagnostics   └─ scope.io             │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  modules/  (image_save, coord_transformations,  │
│             composite_builder, autofocus, …)    │
└─────────────────────────────────────────────────┘
```

Each layer wraps the one below. Higher = easier. Lower = more control.

For internal serial-protocol details (firmware updates, bring-up tooling), see **Appendix A** at the end of this document — not intended for application integration.

---

## Integration Levels

Pick the layer that fits your use case:

| Layer | Interface | Language | Best for |
|---|---|---|---|
| **REST surface** | HTTP (JSON) | Any | External apps, cross-language control |
| **ScopeSession session layer** | Python | Python | Headless scripts, automation, tests |
| **Lumascope + sub-APIs** | Python | Python | Full hardware control, custom applications |

The remainder of this document is organized as the sub-API reference (one section per sub-API), then the modules layer, plugin platform pointers, REST surface, and finally practical patterns + appendices.

### Sentinel-return vs raise contract

Methods on the L2 surface follow one of two contracts; if a method's docstring has a `Raises:` section it follows the raise contract, otherwise the sentinel contract.

- **Hardware-state queries** (capability probes, status reads, getters like `get_led_state`, `get_target_position`, `get_led_states`, `max_gain_db_cached`, `read_motor_fan_rpm`) return a sentinel value -- `None`, `False`, or an empty container -- when the value cannot be read (no hardware, channel not set, firmware does not implement the probe). No exception is raised. The caller branches on the sentinel.
- **Camera value getters** (`get_gain_db`, `get_exposure_ms`, `get_width`/`get_height`, `get_binning_size`) are a stricter subclass of the sentinel contract: a **transient read failure is invisible** -- the getter answers with the validated last-known-good value, so a momentary USB/SDK glitch can never hand you a failure code where a physical value belongs (no `-1` gain into arithmetic, no `None` frame size into a subscript). The documented camera-absent defaults (`get_gain_db` -1.0, `get_exposure_ms` 0.0, width/height getters 0, `get_binning_size` 1) occur **only** when no camera is active or the value has never been successfully read -- stable states you can see coming via `camera_connected`, not something a transient failure produces mid-session. Callers that must record what the hardware was at a specific moment (file metadata, logs of record) use `get_live_camera_settings()` instead: it returns only fields whose driver read succeeded right now (`gain_db`, `exposure_ms`, `frame_size`, `pixel_format`) and omits the rest -- there, unknown stays unknown by design.
- **Naming convention -- `*_cached` vs `get_*`**: a property ending in `_cached` (`gain_db_cached`, `exposure_ms_cached`, `frame_size_cached`, `pixel_format_cached`, `active_cached`, `min_frame_size_cached`, `max_exposure_ms_cached`, `max_gain_db_cached`, `min_exposure_ms_cached`, `min_gain_db_cached`) reads the host-side camera cache and performs **no driver I/O** -- safe to read at any frequency from any thread. A `get_*` method is a **live driver read** under the last-known-good contract above. The name carries the contract, so a call site's I/O behavior is visible without opening the implementation.
- **State-changing operations** (setters like `move_absolute`, `led_on`, etc.) typically return `True` on success and `False` for "couldn't do it" (no driver, mode invalid, driver does not implement, etc.). A `Raises:` section in the docstring documents the typed exception (`HardwareError`, `CaptureError`, `ConfigError` from `modules.exceptions`) that propagates when the underlying SDK call itself fails. Some members still log and show their own notification before re-raising; they are being moved to raise only, so the failure is shown once, by whoever reports it. The typed exception is what L2 callers should catch. **Read the member's own `Raises:` section rather than this paragraph: it is the declaration, and not every setter returns a status.**
- **Camera setting applies** (`set_gain_db`, `set_exposure_ms`, `set_black_level`, `set_frame_size`, `set_binning_size`, `set_pixel_format`) are the raise contract, not the True/False one. A confirmed driver rejection raises `CameraSettingRejected` (`modules.exceptions`), a fault carrying `setting`, `requested`, and the `title` and message a person reads; nothing is logged or shown before it reaches you, so reporting it is yours. Success is observed by the returned value, which is what the camera actually applied and may differ from the request: the DELIVERED size for the geometry setters, and for `set_gain_db` / `set_exposure_ms` the gain in dB / exposure in ms now in effect (a body snaps or quantizes). A gain or exposure outside the camera's declared range (`min_gain_db_cached` .. `max_gain_db_cached`, `min_exposure_ms_cached` .. `max_exposure_ms_cached`) is refused before anything reaches the camera with `CameraSettingOutOfRangeError`, a refusal and a `ValueError`, carrying `reason` (`gain_db_out_of_range` / `exposure_ms_out_of_range`), `requested`, `minimum` and `maximum`, and a message naming the range; an end the camera does not declare is `None` and is not checked. A frame width or height outside [`min_frame_size_cached`, the scope's maximum / the current binning] is refused the same way from `set_frame_size` (`frame_width_out_of_range` / `frame_height_out_of_range`); the scope's maximum is `capabilities.camera_max_frame_size`, unbinned: the sensor, or the model's smaller one where its lens images less of the sensor (1700 on the LS560). When it is `None` the maximum is unknown and is not checked. Inside the range every camera delivers exactly the size asked for: it acquires the next window up on its own grid and crops back. The Session's `set_frame_size` floors the size to even sides first (`get_pixel_alignment()`, `{2, 2}` on every camera) and refuses before it stores. Two cases are deliberately **not** rejections and do not raise: no camera is active (a quiet no-op per the missing-hardware contract), and a driver with no confirmation signal (it answers `None`, meaning "cannot confirm", not "refused"). Gain and exposure rejections are both confirmable on Basler and IDS bodies. On the Classic (FX2) body the sensor register write raises out of the driver rather than reporting a refusal, so a failed apply there reaches you as that exception instead of as `CameraSettingRejected`.
- **Hardware-command dispatch** (LED, motion, and camera commands): each command submits to one of the scope's own lanes -- IO for LED and motion, CAMERA for camera -- and blocks until the hardware has it. Every `Lumascope` builds and starts its two lanes, a bare `Lumascope()` in a script included, so commands from every caller run one at a time per bus, in order. While a protocol run owns the lanes (or a lane is disabled), the command raises `HardwareCommandRefusedError` (`modules.exceptions`), carrying the machine-readable `reason` (`exclusive_activity_running`) and the refused member. After `scope.disconnect()` the lanes are shut, and every command raises at once with `reason` `scope_disconnected`, `stop_motion` included. A motion or LED command for hardware the scope does not have raises it too, before anything moves or lights, and its `missing` (a `MissingPart`, `modules.exceptions`) names the part, its words coming from it: `reason` `not_connected` when the model has a motor controller and none is connected -- never came up, or its cable pulled -- (`MissingPart.MOTOR_CONTROLLER`, "Check the USB cable ..."), or when no LED controller is connected (`MissingPart.LED_CONTROLLER`); `axis_absent` when the scope has no such motor or LED: `MissingPart.X`, `Y` or `Z` for an axis it lacks, `MissingPart.TURRET` for a turret, and `MissingPart.MOTORS` on a manual scope (LS620, LS560), for every motion command, `stop_motion`, `home()` and the acceleration limit included; `MissingPart.led(...)` for an LED the model lacks, named as the command named it ("This microscope has no Red LED.", "... no LED on channel 2."). A name that is no axis at all stays a `ValueError`. An LED off whose end state already holds is not refused: an off of a channel the API holds dark, of an LED the model lacks, or with no LED controller and nothing believed lit returns with nothing written and nothing logged. There is one form of each command; a caller that must not wait runs it on its own thread.
- **Sentinel-return methods log** at `logger.warning` or `logger.info` per Rule 5; they do **not** fire user notifications (no actionable failure occurred -- the value is just unknown).
- **`camera_connected` is an instantaneous, non-latching poll.** A `False` can be transient (a single flaky connectivity query on an otherwise healthy camera). Consumers may skip work on `False` and re-poll on their next cycle; they must never latch, self-cancel, or tear anything down on it -- one transient `False` on a multi-day run should cost one skipped cycle, not the rest of the session.
- **`scope.imaging.camera_removed` is the latched answer.** It is True once the camera's driver has declared it removed (unplugged, off the bus), and stays True until the camera connects again. Unlike a `False` from `camera_connected`, it is never transient, so it is safe to act on: a run that fails a capture after the camera was removed ends at once, `status='failed'`, `reason='hardware_disconnected'`. False on a scope with no camera at all.

If you are writing a new wrapper, the `Raises:` section is the canonical declaration of which contract applies.

---

## Lumascope composition root

The `Lumascope` class is the **hardware-composition-root**. It constructs and holds the eight sub-APIs (`scope.motion`, `scope.illumination`, `scope.imaging`, `scope.diagnostics`, `scope.capabilities`, `scope.runtime_state`, `scope.protocols`, `scope.io`), wires them together, and owns lifecycle (connect / disconnect / emergency shutdown). `scope.protocols` holds the two `Protocol` constructors, documented under Running protocols.

**When to use directly:** you need fine-grained control beyond ScopeSession, or you're building a custom application. The GUI, ScopeSession, and REST surface all go through this class.

### Initialization

```python
from modules.lumascope_api import Lumascope
from modules.scope_init_config import ScopeInitConfig

scope = Lumascope()                       # real hardware (auto-detect camera)
scope = Lumascope(simulate=True)          # simulated (no hardware)
scope = Lumascope(camera_type='pylon')    # force Basler Pylon
scope = Lumascope(camera_type='ids')      # force IDS
scope = Lumascope(simulate=True, source_path='/path/to/lvp')  # another data folder

scope.source_path                         # the data folder the scope was started on
scope.wellplate_loader                    # the labware catalogue, read from it once
scope.objective_helper                    # the objective catalogue, read from it once
scope.scope_models                        # the model catalogue (scopes.json Models), read from it once
scope.scope_models['LS850T']['Turret']    # one model's entry: True, the LS850T has a turret
```

Valid `camera_type` values: `'auto'` (default), `'pylon'`, `'ids'`, `'sim'`.

A scope acts on configuration it does not hold: the labware, the stage offset, the turret map, the objective selected on a scope with no turret, and whether the scale bar is drawn are its session's settings, read whenever it acts on them. A `ScopeSession` binds the scope it composes to its settings. A bare `Lumascope` no session composed refuses those reads with a `ConfigError` saying so, so it serves motion, LEDs and diagnostics; to capture, convert a plate position or draw a scale bar, compose it into a session (`ScopeSession.create(settings, scope=scope)`, or let `create` build it).

```python
scope.objective_helper.get_objectives_list()        # every objective id in the catalogue
scope.objective_helper.get_objective_info('4x Oly') # one entry; ConfigError names an unknown id
scope.objective_helper.get_objectives_dataframe()   # the catalogue as a pandas DataFrame
scope.wellplate_loader.get_plate_list()             # every plate name in the catalogue
scope.wellplate_loader.is_known_plate(name)         # True for a catalogue name or a retired alias
scope.wellplate_loader.resolve_plate_key(name)      # the catalogue's spelling; ConfigError if unknown
scope.wellplate_loader.get_plate(plate_key=name)    # a WellPlate built from the catalogue entry
```

`source_path` is the folder holding `data/`; without one the scope uses the installation's own. The scope reads `data/labware.json`, `data/objectives.json`, `data/scopes.json` and `data/motorconfig_defaults.json` from it once, first, before anything starts, and everything that asks about plates, objectives or models reads the scope's copy: the scope's runtime state, protocol construction and validation, autofocus, the run, the session and the GUI. The model catalogue (`scopes.json`'s `Models`) is `scope.scope_models`, a read-only mapping of model name to its entry. A file that is missing, unreadable or not the shape its reader needs (for `scopes.json`, one with no `Models` section) raises `InstallationFileError` (`modules.exceptions`) naming the file and its folder, and nothing is left running. A simulated scope's `sim_model` that the folder's catalogue does not list raises `ConfigError` before anything starts. The release's layer vocabulary (`scopes.json`'s `LayerOrder`) is the one exception: it is process-wide, read from the installation's own folder the first time any code asks for the layers, and a scope's layers are resolved against it. It is not a `ConfigError`: the installation is at fault, not your settings.

The scope builds and starts its own IO and CAMERA lanes at construction and shuts them in `scope.disconnect()`. `ui_dispatcher=` (`Clock.schedule_once(func, dt)`'s shape) is where the lanes hand a finished command's callback; leave it `None` (the default) and callbacks run on the lane's worker. A GUI host passes its UI marshaller so they reach its UI thread.

### Layer identity

Every scope resolves its **layer identity** at construction — what the layers on this unit ARE: stable key name, display name, LED board address, excitation wavelength, plus the unit's filterset. It resolves from the unit's own configuration when one exists, else from the model's entry; a motor-reported model always outranks `configured_model` (hardware truth beats a selection), and `configured_model` serves models whose hardware cannot report one (the Classic/FX2 line). A simulated scope reports the model it was DECLARED with, so the simulated motor board below reports `LS560` and the layers resolve exactly as that model's catalogue entry says.

```python
scope = Lumascope(simulate=True, configured_model='LS560')

identity = scope.layer_identity          # immutable snapshot
identity.source                          # 'motorconfig' | 'scopes' | 'unresolved'
identity.model                           # the model it resolved as: the board's report, else configured_model; None if neither
identity.filterset                       # the unit's filterset identity string
for layer in identity.layers:
    layer.key_name                       # stable name: settings keys, protocol Color, filenames
    layer.display_name                   # what the operator sees (e.g. 'BF-Phase' on FX2)
    layer.led_channel                    # tuple of board addresses; () = drives no LED
    layer.excitation_nm                  # excitation wavelength; None for broadband/LED-less

identity.find('BF')                      # record by stable key name, or None

# override_model resolves AS another model for this call only
# (lab/testing use — never persisted, and the capabilities do not follow).
scope.refresh_layer_identity(override_model='LS850T')
```

A scope's model is fixed for its lifetime: `identity.model`, `scope.capabilities.model` and the model every saved image records are one answer. A model the catalogue lacks is still carried, with no layers. To change the model of a scope that cannot report its own (the FX2 line), save the selection through the session; it applies the next time the scope is brought up, and a motor board that reports its own model still wins then:

```python
session.select_model('LS560')   # saved to settings['microscope']; raises ScopeModelUnknownError if the catalogue lacks it
session.model_at_next_start     # 'LS560' until the next bring-up; None when the saved model is the one running
```

A scope with no resolvable identity carries the empty `'unresolved'` snapshot: LED commands then raise a named error rather than guessing. Names accepted by `scope.illumination` are the `key_name` values.

Then apply runtime configuration (frame size, objective, binning, stage offset). A Session-built session does this for you: `ScopeSession.create` runs `session.configure_scope()` before it returns (see "ScopeSession session layer"), and that is the form an L2 caller reaches for. The manual form below is for a `Lumascope` you constructed yourself and handed to `ScopeSession.create(settings, scope=scope)`, which binds it to the settings it reads. `ScopeInitConfig.from_settings(settings, scope_config=..., turreted=...)` reads from your LVP settings dict and raises `ConfigError` naming the key when `frame`, `binning`, `stage_offset`, `turret_objectives`, `scale_bar.enabled` or `motion.acceleration_max_pct` is missing, or `objective_id` on a scope with no turret. `turreted` is required and has no default: on a turreted scope the objective is the one assigned to the slot in the light path. The labware, offset, turret map, objective and scale bar are not in the config: the scope reads them from the settings. `initialize` refuses a stored `objective_id` the catalogue does not hold, on a scope with no turret. You can also construct one directly:

```python
config = ScopeInitConfig(
    turreted=False,                  # True on a turret model
    preferred_turret_slot=None,
    binning_size=1,
    frame_width=3840,
    frame_height=2160,
    acceleration_pct=100,
    image_mode='8bit',
    # expects_motion / expects_led default to True; override for
    # models that legitimately have no motor / no LED (e.g. LS620
    # has no motor, so expects_motion=False avoids a spurious
    # "Partial Hardware Detected" popup).
)
scope.initialize(config)
```

### Connection

```python
scope.are_all_connected()                 # LED + motor + camera all up (motor only if expected)
scope.unconnected_parts()                 # ('LED controller', 'motor controller', 'camera') -- those not up; () when all are
scope.motor_connected                     # motor board (property)
scope.motion_expected                     # False on a manual scope (LS620, LS560): no motor board to connect
scope.led_connected                       # LED board (property)
scope.camera_connected                    # camera (property)
scope.imaging.camera_removed              # True once the camera was declared unplugged (latched until it reconnects)
scope.no_hardware                         # True if all-null (no real hardware found)
scope.disconnect()
```

### Objective management

The objective sets the pixel size stamped into every capture, so the Session owns it: whether it is unknowable, how it is confirmed, and the plain writers. On a scope with a turret, the active objective is the one assigned to the slot in the light path, derived on every read: a turret move changes it, and no copy of it is stored beside the slot map. With no turret, it is the selected objective, and `select_objective` moves the settings store and the scope's runtime state together. Every writer refuses an id that is not exactly a catalogue key with `ConfigError`, before any write. The resolved optics (`[Optics   ] objective=... -> N um/px`) are recorded in the log once each time the active objective changes -- after a selection, an assignment or a turret move -- the next time it is read, so before any capture stamps it.

When no one can say which objective is in the light path, it is unknown, never a stored guess: on a turreted scope, before the turret has been homed or moved since bring-up, when its slot has no assignment, or when the assignment is not in the catalogue; with no turret, before anything was selected. `scope.runtime_state.resolve_current_objective()` then raises `ObjectiveUnknownError` (a `ConfigError`; `.reason` is `'slot_unknown'`, `'slot_unassigned'`, `'not_in_catalogue'`, `'none_selected'` or `'turret_undecided'` -- a bare `Lumascope` before `initialize()` has recorded whether it has a turret, and `.slot` is the slot in the light path or `None`), and `get_current_objective_id()` / `get_current_objective()` return `None`.

The plate decides every well position the program computes, so the Session owns it on the same terms: `select_labware` moves the settings store and the scope's runtime state together, or moves neither. It refuses with `ConfigError` -- before either store is written -- a name that is not a string, a name the labware catalogue cannot resolve, and settings with no usable `protocol` block to hold the selection. Plate names that were renamed still resolve, so a protocol saved under an old name is accepted rather than refused.

Its return value reports whether the stored NAME changed, not whether the plate did: selecting a renamed plate under its old name while the new name is stored returns `True` and both names refer to the same plate. Both stores are written on every accepted call, including one that reports no change -- the settings key is not evidence about what the scope holds, so a caller that writes it first cannot make the selection skip itself.

A scope with no XY stage is on "Center Plate": it has one field and no wells to move between, so bring-up replaces any other stored plate with it and logs the replacement, and every protocol made from the settings is born on it.

A protocol has its own plate, and every position a step holds is stated against it. `session.set_protocol_labware(protocol, plate_key)` puts `protocol` on `plate_key` and returns the catalogue key it took; the scope's plate is `select_labware`'s and does not move. A plate the catalogue does not have raises `ConfigError` and the protocol keeps its plate; a renamed plate is stored under its current key. On a scope with no XY stage the protocol takes "Center Plate" whatever was asked, and a different plate asked for is logged as replaced. LumaViewPro's plate list puts the scope and its protocol on the plate picked, the protocol only once the scope has taken it. The underlying call is `scope.protocols.set_labware(protocol, plate_key)`.

```python
session.set_protocol_labware(protocol, '6 well microplate')   # '6 well microplate'
session.scope.protocols.set_labware(protocol, '6 well microplate')   # the underlying call
```

A protocol loaded from disk is put on its own plate by one Session member: `session.load_protocol(file_path)` loads through `scope.protocols.load_protocol` and selects the plate the file names through `select_labware`, and raises what either raises. A refused selection -- a run, a diagnostic or a recording holds the scope and the file names another plate -- refuses the whole load, and the scope stays on the plate it had. On a scope with no XY stage the protocol takes "Center Plate", through `set_protocol_labware`'s rule. The member sets the plate only: the protocol's period, duration and per-layer settings stay in the protocol, and the settings' stored schedule is not changed. To take a protocol's per-layer settings into the layer controls as well, call `session.apply_layer_settings(protocol)` after the load, as LumaViewPro's own Load does: every layer stops acquiring and stimulating, then each layer the protocol names takes its acquire mode and every value its row holds; a blank value leaves that control as it was, and a layer this release does not know, or this scope does not have, is logged and dropped. A script that loads a protocol only to run it does not need it. `protocol.layer_settings()` returns the rows typed: `Acquire` `'image'` or `'video'`; `Illumination`, `Gain` and `Exposure` floats; `Sum` an int; `Auto_Gain`, `False_Color` and `Stim_Enabled` bools; a blank cell None. A file whose Layer Settings block has a cell of the wrong type, or no `Layer` column, is refused at load with `ProtocolFormatError`, naming the file; a file with no block has its layer settings inferred from its steps: each layer with steps acquires (`'video'` when any of its steps records video), and a cell its first step cannot supply is None. A step value the run cannot use is reported by the step check -- a notice at load, a refusal when a run starts -- in a file with a block or without.

```python
question = session.objective_question()            # None, or ObjectiveQuestion(turret_position, proposed, choices)
if question is not None:
    session.confirm_objective(question.proposed, turret_position=question.turret_position)

session.select_objective('10x Oly')                # True when the objective changed; False for the one held.
                                                   # On a turret scope it assigns the slot in the light path;
                                                   # ObjectiveUnknownError while that slot is unknown
session.select_labware('384 well microplate')      # True when the plate changed; a retired spelling is accepted and stored under its catalogue key; both stores are written either way
session.assign_turret_objective(2, '10x Oly')      # slot 1-4 (ValueError otherwise)
session.clear_turret_objective(2)
session.clear_current_turret_objective()          # clear the slot in the light path;
                                                   # ObjectiveUnknownError while that slot is unknown
```

`objective_question()` is a read: it returns a question when no one has confirmed the objective on this install, or when the slot in the light path on a declared turret model has no assignment. The question names the live slot (`scope.motion.get_turret_slot()`) and proposes only that slot's assignment; an unknown slot alone -- as during every turret move -- owes no question, but on an install whose objective has never been confirmed an unknown slot raises `ObjectiveUnknownError` (`.reason` `'slot_unknown'`) rather than ask about a slot the turret may not be in. It may log one withheld-question line per call while a question is owed and suppressed (no hardware; provisional settings) -- a caller that polls it will see that line per poll. A configured session may still have a question to ask: the factories do not ask it. A turret move onto an unassigned slot does not ask either, so a headless caller asks `objective_question()` after the move.

Labware / turret-config / stage-offset are runtime-mutable microscope configuration (not live hardware). The `scope.runtime_state` sub-API answers them, reading the session's settings, their one store, on every answer; it holds no copy and has no setter. L2 callers reach it through the composition root the Session exposes: `session.scope.runtime_state.*`. They are changed only through the Session: `session.select_objective`, `session.assign_turret_objective` / `session.clear_turret_objective`, `session.select_labware`, and `session.update_settings('stage_offset.x', ...)`. What a getter returns is a copy: changing it changes no setting.

```python
scope.runtime_state.is_turreted()                      # True when the objective is derived from the turret slot
scope.runtime_state.resolve_current_objective()        # (id, info), or raises ObjectiveUnknownError saying why
scope.runtime_state.get_current_objective_id()         # None when unknown
scope.runtime_state.get_objective_info('10x Oly')      # {focal_length, magnification, NA, ...}
scope.runtime_state.get_available_objectives()
scope.runtime_state.get_current_objective()            # None when unknown

# Turret integration
scope.runtime_state.get_turret_config()                # {1: '4x Oly', 2: '10x Oly', 3: None, 4: None}, a copy
scope.motion.get_turret_position_for_objective_id('10x Oly')   # returns 2 (turret position is motion state)
scope.motion.is_current_turret_position_objective_set()        # False when the CURRENT turret slot has no configured objective

# Labware + stage offset -- the plate-coordinate inputs
scope.runtime_state.get_labware()                      # the plate the settings select, from the catalogue
scope.runtime_state.get_stage_offset()                 # {'x': ..., 'y': ...} in um, a copy
scope.runtime_state.get_well_label()                   # 'A1' for the current stage XY; '' when the labware has no wells

# Stage µm → plate mm using the selected labware + stage offset
# (the bound form of CoordinateTransformer.stage_to_plate; raises
# ConfigError on a scope no session bound)
px, py = scope.runtime_state.stage_to_plate(sx=60000, sy=40000)

# The same transform bound to copies of the labware and offset selected
# now: a later change moves none of the positions it converts.
to_plate = scope.runtime_state.plate_transform()

# Plate mm → stage µm, one axis. The completing half of stage_to_plate.
sx = scope.runtime_state.plate_to_stage_axis(axis='X', plate_mm=50.0)
```

To MOVE to a plate coordinate, pass it to the motion API in that frame
rather than converting first -- the API checks it against what the stage
can actually reach and refuses in plate mm, naming the coordinate you
asked for:

```python
scope.motion.move_absolute('Y', 180.0, frame='plate')
# PositionOutOfRangeError: Y plate position 180.0 is outside the
# reachable range 1.48 to 81.48.
```

---

## ScopeSession session layer

GUI-free session container. All hardware commands route through executor threads for thread safety. Use this for scripts and automation.

**When to use:** You want to write a Python script that controls the microscope without the GUI.

### Setup

For **real hardware** with settings loaded from disk:

```python
import modules.settings_init as settings_init
from lvp_logger import logger
from modules.scope_session import ScopeSession

# Takes a logger and the appdata DIRECTORY; reads data/current.json
# itself (settings.json is the corrupt-file fallback + defaults-merge
# source) and populates the module-global settings dict.
settings_init.load_lvp_settings(logger, '.')
session = ScopeSession.create(settings=settings_init.settings, source_path='.')

session.source_path                       # the scope's data folder (source_path above)
session.wellplate_loader                  # the scope's labware catalogue -- the same object
session.objective_helper                  # the scope's objective catalogue -- the same object
```

`source_path` is the data folder `create` builds the scope on; leave it out and the scope uses the installation's own folder. The session's folder and catalogues are always its scope's, never copies, so nothing in one session can disagree about which plates or objectives exist.

The session comes back **configured** and **running**: `create` builds the scope, runs `session.configure_scope()` (turret slot keys normalized, the stored objective selected on a scope with no turret, labware selected, `scope.initialize(...)` applied), releases the camera start gate — so `save_image` works without a further `initialize` — and starts the executor lanes: the scope builds and starts its own IO and CAMERA lanes, and the factory builds and starts the FILE lane, the worker pool and the protocol thread around them. Each lane is started once, by whoever built it; starting a running lane again raises `RuntimeError`. `session.shutdown()` is the teardown for everything the factory built (see "Cleanup"): lanes and their threads down, LEDs off, motion stopped, scope disconnected.

`create` takes the host's injections as named keyword arguments; every one of them is optional, and the headless form passes none of them.

```python
session = ScopeSession.create(
    settings=settings_init.settings,
    source_path='.',
    simulate=False,                         # True builds a simulated scope instead of opening hardware
    ui_dispatcher=None,                     # host UI marshaling, Clock.schedule_once(func, dt)'s shape;
                                            # None runs executor callbacks inline on the worker (headless);
                                            # refused beside scope= -- pass it to Lumascope(...) instead;
                                            # so is source_path: a session's folder is its scope's
    af_ui_update_func=None,                 # (pos) -> None, called as autofocus moves Z; None for headless
    settings_saved_hook=None,               # hook(settings_snapshot: dict) after a successful save_settings
    engineering_mode=False,                 # stored on the session
)
```

`af_ui_update_func` is one callable with two consumers: the autofocus runner's Z readout and the capture engine's. There is a seventh parameter, `display_ctx_provider`, which exists for the Kivy host's display thread and is not an L2 parameter — leave it unset.

If you hand `create` a scope you built yourself (`scope=...`), that scope is your bring-up: call `session.configure_scope()` and `session.scope.imaging.start_streaming()` yourself. Its lanes marshal callbacks through the `ui_dispatcher` you built it with, so `create` refuses a `ui_dispatcher` beside `scope` with `ValueError`. One session per scope: a second `create(scope=...)` over a scope a live session holds raises `RuntimeError`.

```python
session = ScopeSession.create(settings=settings_init.settings, scope=my_scope)
session.configure_scope()                 # your scope, your bring-up; may rewrite settings['microscope']
session.scope.imaging.start_streaming()
```

`configure_scope()` asks the motor board which model it is before anything else. When the board reports a model the catalogue (`scopes.json` `Models`) knows and it differs from `settings['microscope']`, the reported model is WRITTEN into the settings dict you passed and logged — hardware truth outranks the stored selection, so your dict can come back changed. A reported model outside the catalogue, or none at all (no motor board), leaves the stored model alone. The scope is yours, so the disconnect is yours too: `session.shutdown()` will not touch a scope it did not build.

To hear what bring-up reports (a camera not found, a partial-hardware warning), pass your outcome listener to the factory: `ScopeSession.create(..., outcome_listener=on_outcome)` registers it before the scope is built. See "Outcomes" below.

**Settings a factory needs.** A file-sourced dict (the loader above) is validated by name and complete. `configure_scope()` adopts the model the hardware reports into `settings['microscope']` whenever the catalogue knows that model, so the microscope key is an input the bring-up may correct. A hand-built dict must carry `frame` and `binning`, and on a scope with no turret `objective_id` -- `configure_scope()` raises `ConfigError` naming the missing key -- and that `objective_id` must name a shipped objective (`data/objectives.json`), or the raise names the objective. The stored plate (`settings['protocol']['labware']`) must be one the labware catalogue has, or `configure_scope()` raises `ConfigError` naming it and the plates available; no other plate is substituted, since a different plate's geometry would put every well position in the wrong place. A turreted scope does not read the stored `objective_id`: its objective is unknown until the turret is homed or moved to a slot, then it is that slot's assignment. `turret_objectives` keys may be JSON strings or ints; the factory normalizes them. A configured session may still owe the objective question (`session.objective_question()`, above); the factories do not ask it. A missing or unusable `labware.json`, `objectives.json`, `scopes.json` or `motorconfig_defaults.json` stops the scope's construction with `InstallationFileError` (see "Initialization").

For **simulated** (no hardware needed, development / CI):

```python
from modules.scope_session import ScopeSession

session = ScopeSession.create(ScopeSession.load_user_settings('.'), simulate=True)
```

`simulate=` is the one choice between simulated and real hardware, the same for every host: `simulate=True` wires up simulated drivers, `simulate=False` (the default) finds the real ones. Either way the factory configures the scope from settings and releases the start gate, so the session it returns can capture and save. `ScopeSession.load_user_settings(source_path)` reads the user's configuration the way the GUI does (`current.json`, then the shipped template); `source_path` must be an LVP installation root (a `data/` directory with `settings.json`), otherwise it raises `ConfigError` naming the root. It checks the settings against the release's layer vocabulary, so an unusable `scopes.json` in the installation's own folder raises `InstallationFileError` naming it. Settings are a required argument of `create`, never read from disk behind your back: pass `load_user_settings(...)` or your own dict. Don't hand-construct a `Lumascope(simulate=True)` + `ScopeSession.create(...)` pair unless you have a specific reason: a bare simulated scope reports the module-global model (`settings['microscope']` when settings are loaded, else `'LS850T'`) unless you pass `configured_model=` yourself, so the bring-up may adopt a model you did not intend.

A simulated scope can also show a stalled camera stream: pass `sim_camera_stall=SimulatedStall(after_s=30, for_s=20)` (from `drivers.simulated_camera`) to `create`, and 30 s after the scope is built the simulated camera stops delivering frames for 20 s while it stays connected and streaming, as a real camera's link can. The GUI takes the same stall as a launch argument beside `--simulate`: `--sim-camera-stall=30,20`. It is refused on real hardware, beside a scope you built yourself (pass it to `Lumascope(sim_camera_stall=...)` instead), and on a model simulated with an FX2 (LS620, LS560), whose camera is not the simulated one.

A simulated session can likewise show a save drive that stops answering: pass `sim_file_stall=SimulatedStall(after_s=60, for_s=120)` to `create`, and 60 s after bring-up the session's file lane is held for 120 s by one write that does not return (`simulated_stuck_write`). A run that ends while its files wait behind it is reported as a stalled file writer (`FileWriterStalledError`) once the write has been stuck for 30 s, with the recovery as its remedy. The GUI takes it as `--sim-file-stall=60,120` beside `--simulate`. It is refused on real hardware and beside a scope you built yourself.

### Application startup sequence

```python
session.start_application_session()                  # home ALL axes, then position turret
session.start_application_session(disable_homing=True)  # skip homing; no startup motion at all
```

`start_application_session()` is the single source of truth for the standard startup orchestration the GUI runs on launch: it queues an all-axis `move_home` on the io_executor (firmware homes Z/T/X/Y in one routine; Z-only boards home what they have), then, when the scope has a turret, moves the T-axis to position 1 (where the home leaves it); the active objective is then slot 1's assignment. Headless / REST callers should use this rather than open-coding the home + turret sequence. `disable_homing=True` skips the home step and, with it, every startup motion: the turret is left where it is, like the stage axes, and no turret position is recorded. Position it yourself after homing.

### Camera capture settings

The image mode, the binning and the frame each have one writer, on the Session: it applies the setting to the camera and stores it in `session.settings` only once the camera took it. A refusal or a rejection leaves the store as it was, so the settings never describe a camera state that is not in force. Call them from the camera's own lane or any thread; each waits for the camera.

```python
session.set_image_mode('12bit_scientific')   # True once stored; ConfigError for an unknown mode;
                                             # CameraSettingRejected if the camera refuses the format
session.set_binning_size(2)                  # the delivered frame {'width', 'height'}; the framed
                                             # region is kept and divided by the new factor.
                                             # CameraSettingUnsupportedError (a refusal) for a size
                                             # the camera does not list
session.set_frame_size(960, 600)             # a displayed (post-binning) size at the stored binning;
                                             # returns what the camera delivered (the size, floored
                                             # to even sides); CameraSettingOutOfRangeError outside
                                             # the scope's range
session.frame_at_binning(2)                  # the frame set_binning_size(2) will ask for; applies nothing
session.get_binning_size()                   # the binning factor in force, e.g. 2: the one the camera took
```

With no camera connected, `set_binning_size` and `set_frame_size` return `None` and store nothing. `set_image_mode` stores the mode for bring-up to apply.

The image mode is a save policy, and every camera honours every mode. `'8bit'` reduces each frame to 8 bits. The three full-depth modes keep the payload at the depth the frame has: `'12bit_scientific'` (labelled "Scientific (full depth)") stores it right-aligned, `'12bit_scaled'` ("Scaled (full depth)") left-justifies it to fill its 16-bit container, and `'12bit_false_color_rgb'` ("RGB (full depth)") writes it as false-colour RGB. The ids keep their `12bit_` names: a camera with a 12-bit format is asked for it in a full-depth mode, and an 8-bit camera (LS560/620/720) delivers 8 bits in every mode. Bring-up runs the saved mode as saved.

A sum (`sum_count` > 1) is stored in a uint16 array on every camera, an 8-bit one included, so a full-depth file keeps every count: four 8-bit frames are a 10-bit sum. Under "Scaled (full depth)" a sum is left-justified like any narrow payload (a 4-sum on an 8-bit camera, tagged 10, is shifted by 6). Under "RGB (full depth)" a summed channel is coloured, and an unsummed frame from an 8-bit camera is written as the 8-bit file it is. A hyperstack whose planes mix 8-bit and 16-bit files is built at 16 bits: under scientific each plane keeps its counts; under scaled and RGB each fills its container, as its own file does.

### Reading and persisting configuration

```python
config = session.get_layer_configs()          # read, in API names
session.update_settings('live_folder', '/data/run7')  # write one setting, from any thread
session.update_settings('video.max_fps', 30)  # a nested setting, by its dotted path
snapshot = session.get_settings_snapshot()     # a consistent copy, taken under the lock
session.get_setting('stage_offset')           # a copy of one setting, by its dotted path; ConfigError when absent
session.scope.settings_template                # every setting there is, with its shipped value
session.set_high_conversion_gain(True)        # the camera takes it, then it is stored; False: neither
session.set_line_noise_reduction(True)        # likewise for the line-noise filter
session.set_scale_bar(True)                   # whether captures draw the scale bar: the setting the imaging API reads
session.set_acceleration_limit(80)            # 1-100, else AccelerationLimitRefusedError (a ValueError); the motors take it, then it is stored; with no motor controller HardwareCommandRefusedError ('not_connected' / 'axis_absent'); nothing stored on a raise
session.save_bookmark(('X', 'Y'))             # the live position as the bookmark: X/Y plate mm, Z um
session.save_all_bookmarks()                  # the live Z as the Z bookmark and every layer's focus
session.save_settings(force=True)             # persist to data/current.json (raises if refused)
session.settings_are_provisional()            # True while current.json is unread and undecided
session.retire_rejected_settings()            # resolve it: retire the unreadable file, saves work again
```

The settings dict IS the configuration surface: its storage keys are the
API names, so what you read is what you write. The session owns the dict
and the lock that guards it. Long-running work should take one
`get_settings_snapshot()` at entry and read from that rather than the live
dict.

**A setting keeps the value asked for when the hardware delivers less.**
An LED current is sent as the board's nearest step: `led_on` returns that
step and the illumination API holds it as the lit current, while the
setting keeps the request. A layer's gain or exposure above the attached
camera's maximum is applied as that maximum
(`scope.imaging.applied_gain_db_for`, `applied_exposure_ms_for`) and the
apply logs both numbers, while the setting keeps the request, so a camera
that can take the value gets it back.

**`update_settings(path, value)` is the one write.** `path` names one
setting by its keys joined with dots (`'BF.sum'`, `'zstack.step_size'`,
`'protocol.filepath'`); the settings that exist, and the kind each holds,
are `session.scope.settings_template`. The write is taken under the lock,
or refused with `SettingRefusedError` (from `modules.exceptions`) and
nothing is written. Its `reason` says why:

| `reason` | Refused when |
|---|---|
| `has_member` | the setting is changed by its own Session member, named in `member`: `microscope` (`select_model`), `objective_id` (`select_objective`), `objective_confirmed` (`confirm_objective`), `turret_objectives` (`assign_turret_objective`), `protocol.labware` (`select_labware`), `image_mode` (`set_image_mode`), `binning` (`set_binning_size`), `frame` (`set_frame_size`), `camera.high_conversion_gain` (`set_high_conversion_gain`), `camera.line_noise_reduction` (`set_line_noise_reduction`), `scale_bar.enabled` (`set_scale_bar`), `motion.acceleration_max_pct` (`set_acceleration_limit`), `bookmark` (`save_bookmark`, `save_all_bookmarks`), a layer's `acquire` (`set_layer_acquire`), `auto_gain` (`set_layer_auto_gain`) and `focus` (`save_focus`) |
| `not_a_setting` | no setting has the path |
| `block` | the path names a block of settings (`'video'`); each is written by its own path |
| `wrong_kind` | the value is not the kind the setting holds: true/false, a number (int or float), text, or a list. A setting shipped as `null` takes any single value. A numpy scalar is refused: convert it with `float()` or `int()` |
| `out_of_range` | `video.max_fps` outside 0 to 200 (0 is no cap); `video.max_duration_seconds` outside 1 to 3600; `tiling_overlap_percent` outside 0 to 50; `image_output_format.live` / `.sequenced` not a format the writer takes; a `live_folder` that is not a path (a NUL byte) |

`protocol.period` (minutes) and `protocol.duration` (hours) are the
schedule a new protocol starts from, held to the protocol's own range
(below): one a protocol cannot run raises `ProtocolScheduleRefusedError`
and nothing is written. A `current.json` written before the range was
enforced can hold one.

At start-up, a stored value the writer would refuse -- the wrong kind, or
outside a range above, a protocol schedule, or a `motion.acceleration_max_pct`
outside 1 to 100 -- is replaced for that key alone by the shipped value, and
the session reports one notice `stored_setting_replaced` as it is created,
naming each replaced setting, its saved value and the one now in its place
(`replacements`: a list of `(path, saved, used)`).
A settings dict handed straight to `ScopeSession.create` holding an
acceleration limit outside 1 to 100 is refused with `ConfigError` before
anything is commanded.

`live_folder` is stored as it is at start-up: a folder given relative to
the installation is made absolute, and the folder is created.

Writing into `session.settings` directly skips every check above and the
lock; it is not a supported write.

`save_settings()` writes the dict to `data/current.json`. **A refused
write raises `SettingsSaveRefusedError`** (from `modules.exceptions`),
carrying a machine-readable `reason` and the refused `file` — a caller
can always tell a refusal from a success. Two reasons exist.
`'no_hardware'`: no hardware was connected during the session, so the
GUI's sliders sit at their defaults and persisting those over a user's
real per-channel values loses them; an API caller has no sliders behind
it, so a deliberate persist passes `force=True`, which overrides this
refusal. `'settings_provisional'`: the app came up on the shipped
template because `current.json` could not be read; that file is the
user's only copy and is left untouched until they decide, so `force`
does **not** override — check `session.settings_are_provisional()` and,
once the user (or the controlling client) has chosen to start over,
resolve with `session.retire_rejected_settings()`. The provisional
refusal protects `current.json` specifically; a save aimed at another
destination still writes. Its `str()` is the sentence written for a
person, and `title` its heading.

### Periodic metrics logging (optional)

```python
session.start_metrics()   # start periodic runtime-health logging
session.stop_metrics()    # stop it (idempotent; shutdown() calls this too)
```

The session owns the metrics-logger lifecycle. Metrics start only when the
host injected a scheduler at construction (`ScopeSession(...,
metrics_scheduler=ThreadingTimerScheduler())` for REST / headless hosts;
the GUI injects a Kivy-clock scheduler) — with no scheduler,
`start_metrics()` is a no-op, so factory-built sessions keep metrics off
unless the host opts in. `settings['profiling']['metrics_interval_s']`
overrides the default hourly cadence. `start_metrics()` raises
`RuntimeError` if metrics are already running; `stop_metrics()` is
idempotent. These members
are host-serialized: call them from one thread (the GUI uses its main
thread only).

### Hardware commands from L2

The Session carries no hardware-command forwarders: every command has
exactly one public spelling, on the sub-APIs of the composition root the
Session exposes as `session.scope`. The dispatch contract (each command
submits and blocks; refusal raises `HardwareCommandRefusedError` while a
protocol run owns the executors) is documented once in the contract
section above.

```python
# LED
session.scope.illumination.led_on('Blue', 200)      # blocks until the write has landed
session.scope.illumination.led_off('Blue')
session.scope.illumination.leds_off()

# Motion
session.scope.motion.home('ALL')
session.scope.motion.move_absolute('Z', 5000)
session.scope.motion.move_relative('X', 500)

# Imaging
session.scope.imaging.set_gain_db(8.0)                 # dB; blocks; returns the dB in effect
session.scope.imaging.set_exposure_ms(50.0)       # ms; blocks; returns the ms in effect
image = session.scope.imaging.capture_and_wait()    # returns frame-valid grab

# A dark frame is never refused. With a channel lit (strictly positive
# current), a frame with no lit pixel is retried until timeout_s in case a
# lit one is coming, then RETURNED with 'dark_saved': True on
# last_capture_info. Read that key to tell a dark capture from a lit one;
# do not re-measure pixels. With nothing commanded -- or a channel at
# 0 mA -- a dark frame is by design and is not measured at all.
# accept_dark=True skips the measurement for callers whose dark frames
# are expected (custom focus sweeps, benchmark probes), so no dark_saved
# fact is filed.
# timeout_s is the retry budget for the content checks (dark floor,
# saturation, chunk verify); leave it 0.0 to judge the first grab only.
# The executor wait is bounded internally.
image = session.scope.imaging.capture_and_wait(timeout_s=2.0)
```

### Capture

`session.manual_capture.capture()` captures one still and saves it, the same
file the GUI's Capture button makes. The caller names only the layer whose
drawer is open (or None), whether it is shown in false colour, and the
overlays it wants; the channel (the lit LED, else that layer, else BF), the
folder (`live_folder/Manual`, per channel when `separate_folder_per_channel`),
the name, summing, format and encoding come from the session's settings. It
returns at once with a `concurrent.futures.Future` of the paths written, the
unmarked file first; an overlay adds a second file from the same frame.
Each file records where the stage was when the frame was grabbed: the plate
position (`PositionX`, `PositionY`, mm) and Z (`PositionZ`, um), each absent
when the scope did not know it (an axis that has lost its reference, or a
scope with no X and Y).

```python
paths = session.manual_capture.capture(
    layer='Blue', false_color_on=True, bullseye=False, crosshairs=False,
).result(timeout=30)
```

It raises `ValueError` for a layer that is not a channel, and
`HardwareCommandRefusedError` with reason `'capture_in_flight'` while an
earlier still is running, `'exclusive_activity_running'` while a run or a
diagnostic holds the scope, and `'scope_disconnected'` when the camera is
closed. When it returns, the still is on the camera lane. The Future raises
`ObjectiveUnknownError` when the objective in the light path is unknown
(nothing captured) and `CaptureError` (reason `'no_frame_returned'`, the
capture engine's cause as its message) when no frame passed.
`session.manual_capture.in_flight` is True from the call until the still
has finished or been refused. A run started while a still is RUNNING waits
for it to finish before it touches the camera, so the still saves under the
state it started with and the run begins after it; a still not yet running
when a run takes the scope (queued behind other camera work) is refused, and
its Future raises `HardwareCommandRefusedError` (reason
`'exclusive_activity_running'`).

To save a frame you already hold, capture it and call `save_image`. It
returns the saved path. When the file cannot be written (a missing or
read-only folder, a full disk) it raises `ImageSaveError`, a `CaptureError`
with reason `'image_save_failed'`, chained from the `OSError`; any other
failure propagates as itself. It shows nothing itself.


```python
from modules.image_save import save_image

# The capture derives the dark-floor expectation itself from commanded
# LED state -- there is no illumination fact to pass.
objective_id, _ = session.scope.runtime_state.resolve_current_objective()  # the objective this frame is taken with
image = session.scope.imaging.capture_and_wait()
save_image(
    session.scope,
    array=image, save_folder='./output',
    file_root='capture', append='_BF',
    channel='BF', false_color_on=False,
    save_encoding='right_aligned',
    significant_bits=session.scope.imaging.capture_frame_depth(image),
    objective_id=objective_id,
)

# Live-view tap: the latest buffered frame, no new exposure forced
# (the capture calls above always force one). (None, None) when unavailable.
frame, timestamp = session.scope.imaging.get_image_from_buffer()

# Payload bit depth of a frame this scope just produced (8, 12, or for a sum
# the bits it can reach) -- needed to interpret or rescale full-depth payloads
# before saving. The capture knows how many frames it summed; pass only the
# frame, and ask before the scope captures again.
session.scope.imaging.capture_frame_depth(image)
# The value that frame saturates at: one frame's full scale, or N of them for
# a sum (not the tag's power of two, which a sum of blown frames misses when N
# is not a power of two).
session.scope.imaging.capture_frame_full_scale(image)
```

### Running protocols

```python
runner = session.create_protocol_runner()
protocol = session.load_protocol('my_protocol.tsv')    # and the scope takes the plate it names
session.apply_layer_settings(protocol)   # optional: its Layer Settings into the layer controls, as the GUI's Load does
# ProtocolFormatError (a refusal) names the file and what is wrong with it -- malformed, too large, or a plate this
# installation's labware catalogue does not have; ProtocolNotLoadedError names a file that cannot be read and the OS reason
# or build one in-memory (config= | input_config= | empty_config=):
protocol = session.scope.protocols.create_protocol(input_config=config)
# save one: whole or not at all. ProtocolNotSavedError (modules.exceptions) names the
# file and the OS reason, and a file already there is unchanged. A run saves its own
# copy in its run folder; when that copy cannot be written the run ends failed_at_start
# (reason run_dir_init_failed) before anything moves.
written = session.save_protocol(protocol, 'my_protocol')   # -> Path('my_protocol.tsv')
# save_protocol adds .tsv to a name without it and writes the protocol with the
# session's Layer Settings block (each layer set to acquire, with the values its
# controls hold), which apply_layer_settings puts back; it returns the path written.
# It does not change what LumaViewPro opens at its next start-up.
# protocol.to_file(path) writes the protocol alone, with the block it was loaded with.

# image_capture_config is REQUIRED: the caller states the run's image mode
# (bit depth + on-disk encoding) explicitly -- there is no silent default.
# Modes: '8bit', '12bit_scientific', '12bit_scaled', '12bit_false_color_rgb'.
pending = runner.run_single_scan(
    protocol,
    image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
)
result = pending.wait(timeout_s=300)
print(result.status, result.reason, result.message)

# Or stop it at any time through the handle its call returned. A handle
# whose run has ended never touches the run that is live now.
pending.stop()
```

**Standalone autofocus.** `run_autofocus(layer)` focuses once on one layer at the current stage position, without a protocol:

```python
pending = runner.run_autofocus('BF', save_characterization_data=True)
result = pending.wait(timeout_s=120)
if result.af_focus_z_um is None:
    print('no focus found; the stage is back where it started')
else:
    print('focused at', result.af_focus_z_um)
if result.af_data_saved:
    print('focus curve written to', result.af_data_path)
```

The layer is named by the caller and has no default: a GUI reads it from whichever drawer is open, which is a fact about a running GUI and means nothing to a script. A layer this release does not have raises `ConfigError` naming it, before any hardware moves. Everything else -- illumination, gain, exposure, frame size, binning, labware, objective -- comes from the settings store, so this run resolves exactly as the standalone autofocus button does.

The run saves no images and writes no run artifacts, and it leaves the stage at the focus it found. A sweep that finds none (a flat or invalid focus curve, an abort, an error) puts the stage back where it started and the run still ends `completed`, because a step's autofocus giving up does not fail a run -- so read `af_focus_z_um`, not the status, to tell the two apart; that is the point of it, so there is no return-to-position input, and a caller sweeping the same field repeatedly sets its own Z between runs. Why a sweep found none reaches your outcome listener (see "Outcomes") as a fault titled `Autofocus Failed` (`AutofocusFailedError`, `modules.exceptions`), whose `reason` is one of three: `flat_focus_curve` (every focus score was zero or invalid: check the sample and the illumination), `out_of_travel` (a window the sweep would search reaches past Z's travel; it is refused before the stage moves into it, and never narrowed to fit, because a focus beyond the travel would come back as the travel's edge: start further from the limit, or narrow the objective's autofocus range), or `unexpected_error` (the sweep raised; the log has the details). An abort is not a failure and reports none. A step's autofocus inside a run reports the same fault, and the step captures at the Z the sweep started from; if that Z could not be put back, the run ends instead of capturing at an unknown position. The standalone run hands the illumination back the way it found it rather than forcing every channel dark, because it is a single-field operation at a scope someone is standing at.

`save_characterization_data` is off by default: a caller that does not ask for the focus curve gets no folder. When on, the data lands under `Autofocus Characterization` in the live folder (or a `parent_dir` you state), and the outcome's `af_data_saved` / `af_data_path` report whether the file was written and where.

**Autofocus all steps.** `run_autofocus_all_steps(protocol)` autofocuses at every step of a protocol and writes the focus it finds into that protocol:

```python
pending = runner.run_autofocus_all_steps(protocol)
result = pending.wait(timeout_s=600)
if result.focus_written:
    print('focused Z:', protocol.steps()['Z'].tolist())
else:
    print('protocol unchanged:', result.status, result.reason)
```

The scan visits each step with autofocus on, whatever the step's own autofocus setting, and captures nothing; the protocol's autofocus settings are not changed. With the `bf_af_for_fluorescence` setting on, a fluorescence step takes the BF step's focus instead of sweeping, as it does in any run. When the scan completes, each step's Z becomes the focus found for it, written before the run lets go of the scope, so a caller reading the protocol in `run_complete` or after `wait` sees the focused values. A step whose sweep found no focus keeps its Z. A scan that does not complete (stopped, failed) writes nothing, because it focused only some of the steps. A protocol whose steps changed during the scan -- a different number of them, or a step at a different position, channel or objective -- is left unchanged, and the person is told once as `Focus Not Saved` (`FocusNotWrittenError` in `modules.exceptions`, a refusal); the outcome's `focus_written` says which happened. The run-wide settings (illumination, gain, exposure, frame size) come from the settings store, so this run resolves exactly as the protocol panel's Autofocus All Steps button does.

**Standalone z-stack.** `run_zstack(layer)` captures a stack on one layer, around the current stage position:

```python
pending = runner.run_zstack('BF')
result = pending.wait(timeout_s=600)
print(result.status, result.reason)
```

The stack's range, step size and reference come from the settings store, so this run resolves exactly as the z-stack panel's Acquire button does, and an unconfigured stack behaves the same way there as here. The layer is named by the caller, and an unknown one raises `ConfigError` before any hardware moves.

Unlike `run_autofocus`, the slices are the product: the run saves its images (under `Manual/Z-Stacks` in the live folder, or a `parent_dir` you state) and writes its run artifacts. Autofocus is forced off for the step and cannot be turned on -- refocusing at each slice re-centres the very range the stack is sweeping. Stimulation configs are carried onto the step as stored, enabled or not, matching the button.

`return_to_start` is on by default: a stack ends at whichever end of its range it finished on, which is not where the operator was looking, so the stage goes back to the position the stack was centred on. Pass `return_to_start=False` to leave it where the stack ended.

`run_single_scan()` runs one scan; `run_protocol()` runs the full multi-scan protocol. LumaViewPro's own Scan and Run buttons start their runs through these two calls. Both take an optional `run_trigger_source`, the provenance recorded on the run and named in a refusal's `holder_trigger` (default `'api_scan'` / `'api_protocol'`; the buttons pass `'scan'` / `'protocol'`), and an optional `engineering_mode`, whether the run stamps the turret position into its filenames (None, the default, reads the mode the session was built in). Both raise `ConfigError` if `image_capture_config` is omitted, and `ProtocolRunRefusedError` (`modules.exceptions`) when the run is refused before any state is committed -- already running, files still writing, empty protocol, a validation failure, hardware not connected, an axis whose position is unknown (`position_unknown`: the scope is not homed, or a home is still running -- the message names each axis), or a live owner holding the illumination. Its `str()` is the sentence written for a person, and an L2 caller branches on its `reason` / `title` / `message` attributes (they map cleanly to a REST status code or a UI message). A run that could not be checked at all -- validating the protocol, or reading whether the hardware is connected, crashed instead of answering -- is not a refusal: it raises `RunCheckFailedError` (`modules.exceptions`, reason `validation_crashed` or `hardware_state_unknown`), a fault with the crash chained as `__cause__`. Neither commits anything, so neither needs unwinding. See the `ProtocolRunner` source for optional callbacks, image-output config, etc. `run_complete` and `files_complete` reach a callback only once the run has let go of the scope -- it is no longer in progress and holds no claim -- so a callback can act on the scope at once; `run_complete` comes first. A run's files stop draining when its last image write lands, just before `files_complete`, so a new run started from inside `files_complete` is not refused `files_writing`. Under LumaViewPro's GUI both callbacks run on its UI thread; with no GUI they run on the thread that ended the run (the run's own thread, or for `files_complete` the file writer's, when the last write lands after the run ends). A callback may wait on its own run: inside one of the run's own callbacks, `handle.wait()` and `handle.wait_for_files()` return once the run has let go of the scope (a composite's, once its merge settles), without waiting for the callbacks. A callback that starts a new run must not wait on it there: the new run's work queues behind the callback, and the wait runs out its bound. `files_complete` fires once per run, after its last image write lands, as `files_complete(protocol=..., run_dir=..., files=...)`: `files` is `'written'`, or `'incomplete'` when some of the run's images are not on disk -- given up on, never taken by the file writer, failed to save, refused because the save drive was nearly full, or a video step's file that did not finish. It counts images: a capture that produced no image is not a file, and is in the outcome's `captures` instead. When an image is not on disk the person is told once, as the last write lands. A callback written for the earlier two-argument form must accept `files`.

**How a run ends.** Every run that commits returns a handle, the run's one identity: everything a caller asks about its run, and its Stop, go through it. `handle.wait(timeout_s=...)` blocks until the run has ended, no longer holds the scope, and its `run_complete`, when it has one, has run, then hands back its outcome, so the caller's next move or setting needs no second wait and finds whatever its `run_complete` did; it gives back `None` when the bound expires. A run's files can still be writing after it lets go of the scope, and a next run is refused `files_writing` until they land: `handle.wait_for_files(timeout_s=...)` blocks for everything `wait` does, then until this run's images are on disk or given up on, its hyperstack build (when it saves hyperstacks) has finished, and the run has said everything about them -- the report of a lost image, and its `files_complete` -- so a next run is admitted, and returns what became of the images -- `outcome` (`'written'`, or `'incomplete'` when any image is not on disk), `written`, `not_written`, and `not_written_reason` (`write_batch_save_failed`, `write_batch_disk_full`, `write_batch_abandoned`, `write_batch_not_taken`, `write_batch_video_unfinished`; `None` when every image landed) -- or `None` when the bound expires. A hyperstack build's own failure is reported by the build, not here. Either wait made on the thread that delivers the run's callbacks -- the GUI's UI thread -- outside those callbacks raises `RunWaitOnUiThreadError` (`modules.exceptions`), since it would wait on itself; and a wait with no bound under a host that has stopped delivering callbacks (a GUI that has closed) does not return. A call that is refused raises and returns no handle, so there is never a stale outcome to wait on.

The outcome carries thirteen fields. `status` is one of `completed`, `incomplete`, `aborted`, `failed` or `failed_at_start`: `incomplete` is a run that reached its end without every capture it was asked for (`captures_failed`; the images it did capture are saved, and the person is told once), `aborted` is an ending someone asked for (`stopped`; `force_reset` and `shutdown` when the application tears the run down), `failed` is one the instrument imposed (`motion_timeout`, `camera_failure`, `disk_space_critical`, `consecutive_scan_failures`, `video_writer_died` (a video step's writer stopped working, so the run was stopped; the frames already written are on disk), `hardware_disconnected` (a part of the scope was disconnected mid-run; a capture that fails after the camera was declared removed ends the run at once), and `position_lost` -- an axis lost its position mid-run, so the run stopped at once and saved no image from that position; home before the next run), and `failed_at_start` is a run that could not begin after it committed. `reason` is the machine-readable cause, stable enough to branch on; `title` and `message` are the sentences a user reads, and never carry raw exception text. `merged`, `artifact_path` and `merge_reason` describe the composite merge only: a run with no merge reports `merged=False` with an empty `merge_reason`, so `merge_reason` is non-empty only when a merge was owed and produced no file. `af_data_saved` and `af_data_path` describe autofocus characterization data the same way: `af_data_saved` is true only when the data file was actually WRITTEN, and `af_data_path` names it. A run that asked for no data, a sweep that measured none, and a run whose queued write an abort discarded all report `af_data_saved=False` with `af_data_path=None` -- the fields answer "did it land", not "was it requested", so a headless caller never has to go looking on disk to find out. `af_focus_z_um` is the Z a standalone autofocus run chose as focus, or None when it chose none; only `run_autofocus` sets it, and every other run reports None, since a run that autofocuses at several steps has no one focus to report -- `run_autofocus_all_steps` answers in the protocol's Z column instead. `captures` is what the run captured of what it was asked for: `asked` (scans times steps for a run that saves images, 0 for one that saves none), `captured`, and `failed`, one entry per capture that produced no image with its `scan`, `step_index`, `step_name` and `cause`; it is counted once per scan and step, so a scan run again after a transient failure never counts a capture twice. `captures` is `None` only for a run a shutdown settled before it ended. `cleanup_failures` names the steps putting the scope back after the run that did not finish (`Restore LED states`, `Restore camera gain/exposure`, `Return to position`, ...): empty when the LEDs, camera settings and stage were all put back, `None` only for a run a shutdown settled before its cleanup ran. The person is told the same once, as `Protocol cleanup issues`. `Stop autofocus` is an autofocus sweep that did not stop within 30 s of the run telling it to: the run keeps its own ending, the sweep can no longer reach the hardware, and autofocus and new runs are refused `autofocus_running` until it stops; restart LumaViewPro if it does not. `focus_written` is whether `run_autofocus_all_steps` wrote the focus it found into the caller's protocol: `None` for every other run, `False` when the scan did not complete or the protocol's steps changed during it (the protocol is unchanged), `True` when it was written.

The two vocabularies are deliberately separate. A run that aborted names why in `reason` and leaves `merge_reason` empty; a run that reached its end but whose merge produced nothing reports its run status (`completed`, or `incomplete` when a channel failed) with the cause in `merge_reason`. `run_composite()` raises `CaptureError` carrying whichever of the two applies. A merge that raised is posted to the listeners as that exception's own type, kind and title -- a refusal such as `PostProcessingRefusedError` (`no_data`, nothing usable to merge) as a warning, `RunFilesNotWrittenError` under "Run Images Not Written" -- with `merge_reason` its `reason` (`merge_error` when it carries none, posted under "Composite Failed"); a merge that could not start or produced no file is posted as `CompositeFailedError`. A composite missing a channel is told as its own `RunIncompleteError`, after the merge's notice, whether it merged or not. The merge's notice and the shortfall are delivered to the listeners before the outcome settles, so `run_composite()` and `handle.wait()` return after them.

`handle.stop()` only asks: the run ends on its own thread, and `wait` tells you when it has. A Stop through a handle whose run has already ended does nothing to any run. When another run is live it is a refusal with reason `run_not_live`, and `holder_trigger` names the run that is live; when nothing is live it raises `RunAlreadyEndedError` -- not a refusal, since the run simply finished before the stop arrived. The handle also answers, about its own run only: `is_live`; `is_stopping`, True from an accepted Stop until the run's teardown finishes; `run_dir`, the run's folder (`None` for a run that saves nothing or never started); and for a progress readout `step_number` (the step executing now, counted from 1), `num_steps`, `remaining_scans` and `interval` (the scan period), each `None` once the run is no longer live.

A refusal with reason `files_writing` means the previous run's files are still draining -- the previous run's `handle.wait_for_files()` returns once they have landed. Reason `files_writing_stalled` means the file writer has stopped making progress entirely (a wedged write, e.g. an unresponsive save drive); waiting will not clear it. Its message says how many unsaved images recovery would lose, and it names its own answer: `refusal.remedy` is a `Remedy` (`modules.exceptions`) whose `member` is `'recover_file_writer'`, with `confirm_text` and `cancel_text` for an offer (the prompt is the refusal's own `title` and `message`). Every refusal has a `remedy` attribute; it is `None` when nothing answers the refusal but waiting or asking differently. The GUI shows a refusal that carries one as a confirmation, and so can any client. A client need not start a run to hear of a stall: once for each stuck write, the session reports `FileWriterStalledError` (`modules.exceptions`, a fault, reason `'files_writing_stalled'`) through the outcome subscription below, carrying the same `remedy` in the same words. Recover with either:

```python
session.apply_remedy(refusal.remedy)   # takes the remedy a refusal named
session.recover_file_writer()          # the same recovery, called by name
```

`session.apply_remedy(remedy)` is the one door from a remedy's name to an action, and returns that action's answer. It takes only the members the Session offers as remedies (today `recover_file_writer`); any other name is refused with `RemedyUnknownError` (reason `remedy_unknown`), before anything runs.

Recovery is deliberate data loss: the finished run's outstanding images are given up on (they were never going to finish), and a partial file from the stuck write may remain on disk. Returns how many images were given up on. It is refused with `FileWriterNotStuckError` (reason `file_writer_not_stuck`) while the writer is still making progress -- those files finish on their own -- and with `HardwareCommandRefusedError` while a run or a diagnostic holds the scope.

**Canonical entry points.** Build the runner with `session.create_protocol_runner()`. Build the `Protocol` it runs with one of the two constructors on the protocols sub-API -- `scope.protocols.load_protocol(file_path)` (from a `.tsv` on disk) or `scope.protocols.create_protocol(config=... | input_config=... | empty_config=...)` (in-memory). Both resolve `data/tiling.json` from the scope's data folder and judge the protocol against the scope's catalogues, so prefer them over calling `Protocol.from_file(...)` directly (which makes you pass `tiling_configs_file_loc` by hand). From a Session, `session.load_protocol(file_path)` is the same load with the scope put on the protocol's plate.

**Tiling grids.** `scope.protocols.tiling_config()` returns the grids this installation offers, read from the same `data/tiling.json`: `available_configs()` lists the labels a protocol's `tiling` accepts (`'1x1'`, `'2x2'`, ...), and `default_config()` is the one to preselect. It reads the file on each call; a missing or corrupt file raises `RuntimeError`. `protocol.tiling()` names the grid a protocol's steps already carry: `'1x1'` when no step is tiled, the grid's label when the tiles form one on offer, and None when they are tiled in a layout no grid offers. `session.apply_tiling` refuses any grid over a tiled protocol (`ProtocolRunRefusedError`, reason `already_tiled`).

```python
grids = session.scope.protocols.tiling_config()
grids.available_configs()   # ['1x1', '2x2', '3x3', ...]
grids.default_config()      # '1x1'
```

**Choosing what each layer captures.** `session.set_layer_acquire(layer, mode)` sets whether a layer captures an image (`'image'`), a video (`'video'`) or nothing (None), as the GUI's acquire toggle does; the layers set to acquire are the ones `new_protocol` and `add_step` build steps for and a composite merges. A layer set to acquire stops stimulating. An unknown layer or mode raises `ConfigError` and changes nothing, and so does `'image'` or `'video'` for a layer this scope does not have (`scope.layer_identity`); setting any layer to acquire nothing is always accepted. A saved setting that has such a layer acquiring -- one written on another model -- is set to acquire nothing when the session comes up, and the change is logged.

```python
session.set_layer_acquire('BF', 'image')
session.set_layer_acquire('Green', None)
```

**Turning a layer's auto-gain on and off.** `session.set_layer_auto_gain(layer, enabled)` does what the GUI's Auto Gain/Exp box does. Turning it on (`True`) stores the preference only; the camera starts adjusting when the layer is next applied, so a script applies the layer afterwards (`session.apply_layer_camera(layer)`, as the GUI does). Turning it off (`False`) stops the camera adjusting and keeps what it reached: the gain becomes the layer's stored gain and the exposure its stored exposure, rounded to 0.1 dB and 0.01 ms. The stored exposure is never below the channel's usable floor (0.1 ms for brightfield, phase contrast and darkfield, 1 ms for fluorescence and luminescence), even when the camera went lower on a bright sample. A value the camera did not report is left as it was. Turning it off returns the lock result: its `state` says whether the camera converged, hit the exposure ceiling (`MAXED`), hit the floor (`AT_MINIMUM`), or reported nothing usable (`FAILED`); a `state` of None means auto-gain was not running. Turning it on returns None. Turning it off waits for the camera, and raises `HardwareCommandRefusedError` and stores nothing while a run or a diagnostic holds the scope. An unknown layer, an `enabled` that is not `True` or `False`, or turning it on for a layer this scope does not have raises `ConfigError` and changes nothing; turning it off is always accepted.

```python
session.set_layer_auto_gain('BF', True)
lock = session.set_layer_auto_gain('BF', False)
lock.state, session.settings['BF']['gain_db'], session.settings['BF']['exposure_ms']
```

**Creating a protocol.** `session.new_protocol(tiling='1x1', use_zstacking=False, period=None, duration=None)` does what the GUI's New does: one step per layer whose `acquire` is set, at every well of the session's labware, with the current objective, tiled and z-stacked as asked. `period` and `duration` are `datetime.timedelta`s; one left out (None) is the stored default's (`settings['protocol']`), and one scan is `timedelta(0)`. The GUI passes the schedule on screen. When no layer is set to acquire it raises `ProtocolRunRefusedError` with reason `no_acquiring_layer`, and a `tiling` this installation does not offer raises it with reason `tiling_unknown`; each is logged and notified once and builds nothing; a labware with no wells gives an empty protocol to fill with `add_step`. `session.create_empty_protocol()` is the no-step protocol that needs no objective.

**A protocol's schedule.** `protocol.period()` and `protocol.duration()` are the protocol's own, and the run runs them. `protocol.modify_time_params(period=..., duration=...)` sets both: a period is None or `timedelta(0)` (one scan) or at least one second, and a duration is None, `timedelta(0)` or more. Anything else -- a sub-second or negative period, a negative duration, a value that is not a `timedelta` -- raises `ProtocolScheduleRefusedError` (from `modules.protocol`; a `ProtocolFormatError`, a refusal) naming the value and the rule, and the protocol keeps the schedule it had; nothing is raised to one second. A protocol built with one, or a file carrying one, is refused the same way, the file by name. Post-processing a finished run reads its saved protocol without judging the schedule, which it never uses, so a run saved by a release that allowed a shorter period can still be stitched or projected. `schedule_from_units('period', minutes)` and `schedule_from_units('duration', hours)` convert the units the file and the settings hold, refusing what is not a runnable number.

```python
from datetime import timedelta
protocol.modify_time_params(period=timedelta(minutes=10), duration=timedelta(hours=24))
protocol.modify_time_params(period=timedelta(milliseconds=500), duration=timedelta(hours=1))  # ProtocolScheduleRefusedError
```

**Reading a protocol's steps.** `protocol.steps()` returns a copy of the steps table, and `protocol.step(i)` a copy of one row; each row `protocol.layer_settings()` returns is a copy too. The dicts in a step's `Video Config` and `Stim_Config` cells are read-only (`modules.protocol.ReadOnlyDict`, a `dict`): a write into one raises `TypeError`, and `copy.deepcopy(cell)` gives a plain dict to edit. A write to a copy changes nothing in the protocol: a protocol changes only through its writers (`modify_time_params`, `session.add_step`, ...), which judge each change. A run works on its own copy of the protocol it is handed, so nothing the run writes, such as an autofocus's Z, reaches the caller's protocol either, unless the run is asked to write its focus back (`run_autofocus_all_steps`).

**A protocol's capture root.** `protocol.modify_capture_root(text)` stores the root as given, and `protocol.capture_root()` returns it so. `protocol.capture_prefix()` is that root as the prefix of a saved file's name (only letters, digits, `-` and `_` kept): the one prefix a run's images and post-processing's outputs both carry, so `exp/2026:10` names files `exp202610_...`.

**Adding a step.** `session.add_step(protocol, before_step=... | after_step=...)` does what the GUI's Add Step does: one step per layer whose `acquire` is set, at the live stage position read on the protocol's plate (`protocol.labware()`, which need not be the session's selected plate; `session.get_current_plate_position()` reads on the selected one), with the current objective, in the settings' `step_channel_order`. With no `before_step` or `after_step` the steps follow the last step; giving both raises `StepEditRefusedError`. It returns the inserted step names in protocol order. When any axis (X, Y, Z, or the turret) does not know its position -- never homed, homing, or lost after a failed home -- it raises `ProtocolRunRefusedError` with reason `step_position_unknown`, naming the axes: the position read keeps answering the last number an axis reported, so a step saved then would record a place the scope no longer vouches for. When no layer is set to acquire it raises `ProtocolRunRefusedError` with reason `no_acquiring_layer`; on a turret scope whose slot is unknown or has no objective assigned, reason `turret_objective_unset`; when the objective is otherwise unknown (a slot assigned an objective that is not in the catalogue, or `objective_id=None` passed below), reason `objective_unknown`. Each is logged and notified once; nothing is added. The underlying call, for a caller supplying its own inputs, is `scope.protocols.add_step(protocol, layer_configs=..., stim_configs=..., plate_position=..., objective_id=... (None when unknown, which is refused), channel_order=..., before_step=... | after_step=...)`.

```python
protocol = session.new_protocol()                   # one step per acquiring layer at every well; refused no_acquiring_layer when none acquires
protocol = session.create_empty_protocol()          # no steps; needs no objective, so it works before one is known
names = session.add_step(protocol, before_step=0)   # ['custom0000_BF', ...]
```

**Updating a step.** `session.update_step(protocol, step_idx, layer=..., label=None)` does what the GUI's Update Step does: step `step_idx` takes `layer`'s settings, the live stage position read on the protocol's plate, as an add does, and the current objective. `label` renames the step; `None` keeps its label. When `layer`'s stim config is enabled the update is a stim edit, so the step keeps the channel it already acquires. It returns the step's name after the update. It is refused for the same reasons, under the same names, as an add -- `step_position_unknown`, `turret_objective_unset`, `objective_unknown`, and `no_acquiring_layer` when the layer the step takes (for a stim edit, the step's own) is not set to acquire -- each logged and notified once, with the step unchanged; a `step_idx` that is not a step of `protocol` raises `StepNotFoundError`. The underlying call is `scope.protocols.update_step(protocol, step_idx, layer=..., layer_configs=..., stim_configs=..., plate_position=..., objective_id=..., label=...)`.

```python
name = session.update_step(protocol, 0, layer='Blue')   # 'custom0000_Blue'
```

**Deleting and renaming a step.** `session.delete_step(protocol, step_idx)` does what the protocol panel's Delete does: step `step_idx` is removed and the steps after it move up one. `session.rename_step(protocol, step_idx, name)` does what typing in the step's name field does: the step takes `name` as its label, kept through later channel changes, and it returns the step's name after the rename. Characters a filename cannot carry are removed from `name`; a name with no letter, digit, dash or underscore left raises `StepEditRefusedError` and the step keeps its name. A `step_idx` that is not a step of `protocol` -- including any index of a protocol with no steps -- raises `StepNotFoundError`, and nothing changes. The underlying calls are `scope.protocols.delete_step(protocol, step_idx)` and `scope.protocols.rename_step(protocol, step_idx, name)`.

```python
session.rename_step(protocol, 0, 'center')   # 'center_BF'
session.delete_step(protocol, 1)
```

**Tiling and z-stacking a protocol.** `session.apply_tiling(protocol, tiling)` does what the protocol panel's tiling Apply does: every step becomes a grid of tiles, `tiling` one of the labels `tiling_config().available_configs()` lists. The tiles are spaced for the configured frame, binning and tile overlap (`tiling_overlap_percent`), and laid out on the protocol's own plate. `session.apply_zstacking(protocol, range_um=..., step_size_um=..., z_reference=...)` does what the z-stack Apply does: every step not already in a stack becomes a stack of slices `step_size_um` apart over `range_um`, its own Z at the stack's `'top'`, `'center'` or `'bottom'`. Both leave the steps in the order a run visits them, and report the step check's notice as the other edits do. Each is refused before any step changes. `ProtocolRunRefusedError` covers a grid this installation does not offer, a protocol already tiled (reason `already_tiled`), a step's objective not in the catalogue, a range or step size not greater than zero, a scope with no motor for the axes the build moves (reason `positions_unreachable`; a Z-only scope cannot be tiled), and a tile or slice outside the stage's travel. `ConfigError` covers an unknown `z_reference` and a protocol plate the catalogue does not have. The underlying calls are `scope.protocols.apply_tiling(protocol, tiling, frame_dimensions=..., binning_size=..., overlap_percent=...)` and `scope.protocols.apply_zstacking(protocol, range_um=..., step_size_um=..., z_reference=...)`.

```python
session.apply_tiling(protocol, '3x3')
session.apply_zstacking(protocol, range_um=20.0, step_size_um=5.0, z_reference='center')
```

**Saving a focus.** `session.save_focus(protocol, layer, step_idx=None)` does what the layer panel's Save Focus does: the live Z becomes `layer`'s saved focus (what every new step of the layer is born at, `session.saved_focus(layer)`), and, when `step_idx` names a step of `layer`, that step's Z. A step of another channel is left alone, and no other step is written: every step of a layer is born at the same focus, so a step that matches the old focus says nothing about whether it should take the new one. It returns a `SavedFocus` (`modules.scope_session`): `z`, the Z saved, and `step_idx`, the step that took it or None. `session.apply_focus_to_layer_steps(protocol, layer)` does what Apply to Steps does: the live Z becomes the layer's focus and the Z of every step of `layer`, and it returns how many steps took it. Both are refused before anything is written: a `layer` this scope does not have with `ConfigError`; on a scope with no Z axis with `ProtocolRunRefusedError`, reason `positions_unreachable`; when Z does not know its position (never homed, homing, or lost after a failed home) with `AxisStateUnknownError`, each logged and notified once; and a `step_idx` that is not a step of `protocol` raises `StepNotFoundError`. The underlying calls are `scope.protocols.focus_z(then=...)` (the live Z, refused as above), `scope.protocols.set_step_z(protocol, step_idx, z)` and `scope.protocols.apply_focus_to_layer_steps(protocol, layer, z)`.

```python
saved = session.save_focus(protocol, 'BF', step_idx=0)   # SavedFocus(z=5000.0, step_idx=0)
count = session.apply_focus_to_layer_steps(protocol, 'BF')

# The underlying calls, for a caller writing a Z it has read itself:
z = scope.protocols.focus_z(then='save the focus')
scope.protocols.set_step_z(protocol, 0, z)
scope.protocols.apply_focus_to_layer_steps(protocol, 'BF', z)
```

**A step's motor targets.** `scope.protocols.step_targets(protocol, step_idx)` is the one conversion a run and a person's navigation share: a `StepTargets` (`modules.lumascope_api.protocols`) of `turret_slot` (the slot carrying the step's objective, chosen as `scope.motion.get_turret_position_for_objective_id` chooses it; None on a scope without a turret), and `x`, `y` and `z` in stage micrometres, X and Y converted on the protocol's own plate, not the scope's selected one; an axis the scope has no motor for is None, and nothing drives it. They come from `scope.protocols.stage_targets(protocol, px, py, z)`, which answers `(x, y, z)` the same way for any plate position and Z: the one conversion every step move takes, so a run's return to its first step between scans and after a z-stack drives only the axes the scope has too. It raises `StepNotFoundError` for a `step_idx` the protocol lacks and `ConfigError` for a plate the catalogue does not have. `scope.protocols.plate_to_stage(protocol, px, py)` is the plate-to-stage conversion under it, on the protocol's plate, whatever axes the scope has; all three take a `stage_offset=` to convert with a snapshot instead of the live offset, as a run does. `scope.protocols.plate_transform(protocol, stage_offset=offset)` is the inverse, bound when called: a function of `(sx_um, sy_um)` answering `(px_mm, py_mm)` on the protocol's plate with that offset, or None when `stage_offset` is None (a scope with no X/Y stage). A run states every frame it saves through it, stills and recorded frames alike: the position the stage was read at beside the grab, never the step's planned one, in the frame the run's steps were driven in, whatever plate is selected meanwhile; an axis the scope lacks or has lost its reference on states none.

```python
targets = scope.protocols.step_targets(protocol, 2)   # StepTargets(turret_slot=1, x=..., y=..., z=...)
sx, sy = scope.protocols.plate_to_stage(protocol, 60.0, 40.0)
x, y, z = scope.protocols.stage_targets(protocol, 60.0, 40.0, 5000.0)   # None for each axis the scope has no motor for
to_plate = scope.protocols.plate_transform(protocol, stage_offset=offset)   # stage um -> the protocol's plate mm, bound now
```

**Going to a step.** `session.go_to_step(protocol, step_idx)` does what a click on a step does, as one task on the scope's IO lane: it asks every axis once whether it knows its position, turns the turret to the slot carrying the step's objective, starts X, Y and Z towards the step's targets together (`scope.protocols.step_targets`, above), puts the step's values into its layer's live settings (the layer acquiring as the step does, its focus at the step's Z, its stimulation from the step's), and applies the step's LED preview: the step's channel at its current when `protocol_led_on` is set, every channel dark when not. While the stage travels it puts the step's layer on the camera (`session.apply_layer_camera`, below), and it returns once the camera holds that layer's exposure, gain and auto-gain and X, Y and Z have arrived, waiting off the IO lane so the lane takes other work meanwhile; a move that does not arrive raises `MoveNotCompletedError`, with the layer and the preview already the step's, and a setting the camera refuses raises `CameraSettingRejected` once the stage has arrived. `session.start_go_to_step(protocol, step_idx)` does the same but for the camera and returns the started X, Y and Z moves without waiting (their `wait()` gives each outcome): the form for a click, where a second click re-targets the stage at once, and whose caller applies the camera. Only the axes the scope has are moved: a manual scope moves nothing, does the rest and returns no moves; a motorized scope whose motor controller is not connected is refused with `HardwareCommandRefusedError` (`not_connected`) before anything changes, and so is a step whose preview would light on a scope with no LED controller connected (a dark preview needs none, so with `protocol_led_on` off the stage still moves); after `scope.disconnect()` it raises `scope_disconnected`. A repeat of the step the session last went to (a re-click) does everything but the preview, so a channel lit or put out in between stays as it was; a run transition forgets it. It is refused before anything changes: a `step_idx` that is not a step of `protocol` with `StepNotFoundError`; a step whose objective this scope cannot put in the light path with `ProtocolRunRefusedError`; a step on a layer this scope lacks, or a stimulation naming no layer, with `ConfigError`; an axis that does not know its position with `AxisStateUnknownError` (logged and notified once); a scope held by a run or a diagnostic with `HardwareCommandRefusedError`. A step outside an axis's travel raises `PositionOutOfRangeError` from that axis's move; the axes before it have moved and nothing else changes.

```python
session.go_to_step(protocol, 2)                       # turret, X, Y, Z, the layer, the LED, the camera; returns on arrival
moves = session.start_go_to_step(protocol, 3)         # the same, started; each move's wait() gives its outcome
```

**Putting a layer on the camera.** `session.apply_layer_camera(layer)` puts the layer's stored exposure, gain and auto-gain on the camera and waits; it returns the gain and exposure now in effect (None with no camera). A stored auto-gain arms the live auto loop, capped to the layer's channel class; a camera without hardware auto-gain applies the layer manually. Bring-up applies BF this way (skipped with no camera, or on a scope with no BF layer), and `go_to_step` the step's layer. Refused: a layer this scope lacks with `ConfigError`; a scope held by a run (an autofocus is one) or a diagnostic with `HardwareCommandRefusedError`; a setting the camera refuses raises `CameraSettingRejected` with the other settings applied.

```python
session.apply_layer_camera('Blue')                   # {'gain_db': ..., 'exposure_ms': ...} now in effect
```

**A step the run will refuse.** Every step cell is read as its column's type when a protocol is loaded or built: numbers for X, Y, Z, Illumination, Gain and Exposure, whole numbers for Sum, Z-Slice and the group IDs, True or False for Auto_Focus, Auto_Gain, False_Color, Custom Step and Auto_Named, `'image'` or `'video'` for Acquire, and a parsable config for Video Config and Stim_Config. A cell of the wrong type -- text in a number column, `maybe` in a True/False column, an empty position -- refuses the load with `ProtocolFormatError`, naming the file, the step and the column; nothing is guessed for it. A protocol built in memory (`config=`) is refused the same way, and the protocol's own writers refuse a value of the wrong type with `StepEditRefusedError`. A step index the protocol lacks is refused with `StepNotFoundError` by every writer and by `protocol.step(i)`; its words say how many steps there are ("The protocol has 6 steps.") and name no index, which is zero-based where a person counts from 1, and it carries `index`, the index refused as it was given, and `num_steps`. Both are `ProtocolError`s and refusals (`modules.protocol`): the reporter shows them as a warning, under "Step Not Changed" and "No Such Step", with no traceback. A step whose cells are the right type can still hold a value the run gate rejects (an exposure of 0, an objective not in the catalogue, an illumination above the board's maximum), and an add or update composed from live settings can too. Neither is refused: the load and the edit succeed, so the step can be fixed, and `session.load_protocol`, `add_step`, `update_step`, `delete_step` and `rename_step` report one notice, reason `protocol_steps_invalid`, listing the validator's lines. The run start refuses the protocol until every step is valid. A step on a layer this scope does not have is refused at load and when a run starts, reason `layer_not_on_scope`, naming the layers and the steps.

### Video steps and recordings

A protocol step with `Acquire` = `video` records through the session's recording engine for the step's configured duration. This is the supported video path for L2 / headless callers; the GUI's manual Record button is a GUI-hosted convenience on the same engine.

Per recording (one per well per scan), the run produces:

- a frames folder (`<step>_video/`) of per-frame TIFFs, numbered in capture order;
- `recording_manifest.json` in that folder -- the measured truth: delivered frame count, measured frame rate, per-frame timestamps, and the recording's end reason. Downstream consumers (including Create Video's `auto` rate) read the manifest, not the configured rate;
- one variable-frame-rate MP4 per recording;
- when the recording is saved as frames, each frame's TIFF carries its own record of when it arrived: the plate position (`plate_pos_mm`, `x_pos`, `y_pos`) and Z (`z_pos_um`) as the scope tracked them, each absent when the scope did not know it; `stage_moving`, true when any axis was moving or homing at delivery; and `channel`, the channel that lit the frame. These are the same keys a still capture writes, and the frame is false-coloured as its recorded channel;
- after the run completes, one OME-TIFF hyperstack per (well, scan): `T` = frame capture order, `C` = channel, per-plane `DeltaT` from the frames' own timestamps. Hyperstacks build at run completion on every host -- headless and REST runs included, no GUI involved. Nothing waits on that build, so its outcome is told as a notification: "Saving Hyperstacks" as it starts, replaced by "Hyperstacks Saved" with the count and folder, or by what went wrong -- including a build that saved only some of its stacks, which names the ones missing and where the rest are.

A manual frames recording with the hyperstack output format on builds one OME-TIFF hyperstack from its frames, `T` = frame order, with each plane's position from the frame's own record (an axis the scope did not know on any frame is omitted from every plane, as the OME writer refuses to invent one). It is built only when one channel lit the whole recording: the stage and the LEDs stay open to L2 callers while a manual recording runs, and a channel change mid-recording leaves frames no single T x C cube can hold, so the build refuses, the `Hyperstack Not Built` notification carries the builder's reason, and the frames stand with their true per-frame channels.

Rate and duration come from the run's settings snapshot at start: `video.max_fps` (0 = uncapped; the effective rate is measured, not assumed) and `video.max_duration_seconds`. Mid-run settings edits do not affect a run in flight.

Recording starts are guarded like protocol starts: `RecordingRefusedError` (`modules.exceptions`) mirrors the `ProtocolRunRefusedError` shape -- its `str()` is the sentence written for a person -- with machine-readable `reason` codes `recording_active` (another recording is live, or still finishing), `exclusive_activity_running` (a protocol run or other exclusive activity holds the session's activity claim), `camera_inactive`, `camera_exposure_unknown`, `insufficient_disk` and `capture_location_unusable`. Nothing is started when it raises.

While a manual recording holds the scope -- live or still draining its frames -- it refuses, for every caller, what would falsify its file: `imaging.set_frame_size`, `set_binning_size` and `set_pixel_format` (and the Session's `set_frame_size`, `set_binning_size` and `set_image_mode`, before either store is written), `motion.move_turret`, a home that moves the turret (`'T'`, or `'ALL'` on a scope with one), the Session's objective writers (`select_objective`, `assign_turret_objective`, `clear_turret_objective`, `clear_current_turret_objective`) and `select_labware`. Each raises `HardwareCommandRefusedError` with `reason='exclusive_activity_running'` and `holder='recording'`. X/Y/Z moves, a single-axis home, LED, gain and exposure stay open to an API caller; a recording's feed-death bound follows an exposure raised mid-recording. The LumaViewPro GUI locks its whole control surface during a live recording, Record/Stop excepted.

Both refusal errors say busy-with-what: `holder` carries what holds the microscope at refusal time (`'protocol'`, `'recording'` or `'diagnostic'` for the exclusive-activity owner, `'autofocus'` for a sweep that did not stop when its run ended, None for refusals that are not holder-shaped), and `holder_trigger` carries the `run_trigger_source` of the run behind that holder -- the run holding the scope, or, for an `autofocus_running` refusal, the run that dispatched the sweep (`'protocol'`, `'autofocus_scan'`, `'zstack'`, `'autofocus'`, `'api_scan'`, `'api_composite'`, `'composite'`, ...). The two are different axes: `holder` is the activity, `holder_trigger` the provenance string the run's starter passed, so a standalone autofocus refused for its own sweep carries `'autofocus'` in both. A recording holder has no trigger -- its kind is the whole answer. File-drain refusals (`files_writing*`) carry the just-finished run's trigger so a poller can report whose files are draining. The `message` names a run by its kind in words ("The Z-stack run is using the microscope"), never by its trigger token; branch on `holder_trigger`, show `message`.

**Opening hyperstacks in Fiji:** the container is OME-TIFF; channel color travels as OME `Channel.Color`. Open via `Plugins > Bio-Formats > Importer` with **Color mode = Composite** (the choice persists per user through that dialog). A plain `File > Open` renders ImageJ's default LUTs, not the file's channel colors.

**Run-state semantics:** `session.is_protocol_running` (a property, not a call) reports True while a run holds the session's exclusive-activity claim -- protocol runs, single scans, z-stacks, autofocus scans, and the standalone Autofocus button's run included -- and a run started inside `session.diagnostic_claim()`, whose holder stays `'diagnostic'` (so a refusal still names the diagnostic) while the run is live. It releases at run-cleanup end; the short post-run file-drain window (files still writing after the run finished) reads False here and True on `session.run_lockout` / `session.protocol_files_draining`, so a poller that must wait for the disk to settle checks those. A live video recording is not a run: it reads False here and is visible on `session.exclusive_activity == 'recording'`.

### Post-processing

Builds over a folder of captured images are session members, the same ones LumaViewPro's Post-Processing buttons call:

```python
pp = session.post_processing

pp.stitch(folder, mode='quality')            # or 'fast_preview'; one mosaic per tiled group
pp.zproject(folder, method='Max')            # one of ZProjector.methods()
pp.composite(folder)                         # output format and per-channel blend thresholds
                                             # are the user's configuration, as a run's merge uses
pp.video(folder, frames_per_sec=None,        # None: a recording plays at its own measured rate
         timestamp_overlay=False)
pp.enhance(target)                           # Quick Enhance: one image, or every image in a folder;
                                             # derived files are written beside their sources
pp.count_cells(folder, method=method_dict)   # writes results.csv into the folder

def on_progress(percent, text):              # optional on every member: percent done, and a
    print(percent, text)                     # status line or None; called on the build's thread
pp.stitch(folder, on_progress=on_progress)
```

Each member blocks until its build finishes and returns a dict whose `message` says what was made, in words written for a person. It raises `PostProcessingRefusedError` (`modules.exceptions`) when the folder cannot yield the output (no images, no groups the build can combine, or a setting it cannot use: an unknown stitch mode or projection method, a playback rate below 1). It raises `PostProcessingFailedError` when the build did not produce everything asked of it; its `produced_paths` lists what was written. A cell count refuses a folder with nothing it could read before touching `results.csv`, and replaces an existing `results.csv` only with a complete new one.

A cell-count method is a dict; start from `modules.post_processing.default_cell_count_method()` (a new copy on every call; the method LumaViewPro's panel starts from) and change what you need:

```python
from modules.post_processing import default_cell_count_method

method = default_cell_count_method()
method['filters']['area'] = {'min': 20, 'max': None}   # square microns; None is an open bound
pp.count_cells(folder, method=method)
```

Each image is measured at the scale it states: a LumaViewPro capture carries its pixel size, so its areas and perimeters are in square microns and microns. An image that states no scale (a PNG, a capture saved without one) is counted in pixels. `results.csv` has one row per image with `file`, `time`, `num_cells`, `total_object_area`, `area_unit` (`um2` or `px2`, that row's unit) and `total_object_intensity`. Setting `context.pixels_per_um` overrides the scale for every image, for example on external images whose scale you know. The area and perimeter filters are in microns, so an image counted in pixels with either one set is not counted: the count goes on with the rest and raises `PostProcessingFailedError` naming it. `modules.post_processing.cell_count_scale(method, image_pixel_size_um)` returns the pixels per micron a count uses for an image (None for pixels).

| Field | Meaning |
|---|---|
| `context.pixels_per_um` | `None` (the default) to use each image's own scale; or a positive number of camera pixels per micron, which overrides it for every image |
| `context.fluorescent_mode` | `True` for bright objects on a dark background; `False` inverts the image first |
| `segmentation.algorithm` | `'initial'` |
| `segmentation.parameters.threshold` | brightness threshold, percent of full scale |
| `filters.area`, `filters.perimeter` | `{'min', 'max'}` in square microns and microns; open (`None`) by default. A set bound refuses an image counted in pixels |
| `filters.sphericity` | `{'min', 'max'}`, 0 to 1 |
| `filters.intensity.min` / `.mean` / `.max` | `{'min', 'max'}` for each region's minimum, mean and maximum intensity, percent of full scale |

`count_cells` refuses a method the count cannot use before the count is queued: `PostProcessingRefusedError` with `reason='method_invalid'`, its message naming the field (a missing field, a `pixels_per_um` that is not a positive number, a bound that is not a number or `None`, a filter whose `min` is above its `max`). `modules.post_processing.check_cell_count_method(method)` asks the same question without counting. A method saved by LumaViewPro's panel also carries a `metadata` entry; the count does not need it. A method file saved before LumaViewPro read each image's scale (`metadata.version` `'1'`) carries the old fixed default of 1.0 pixels per micron: `load_cell_count_method` drops it, so the method uses each image's own scale, and reports a notice once.

`modules.post_processing.read_cell_count_results(path)` reads a `results.csv` back as a pandas table, its `time` column a datetime column (the count's own time format); `results_axes(table)` returns the columns a graph can take as X (numbers and times) and as Y (numbers). A file that cannot be graphed is refused: `PostProcessingRefusedError` with `operation='Graphing'`, `reason='results_unreadable'`, its message naming the file. `modules.graph_analysis.trendline_kinds(x, y)` names the trendlines two columns take (a time X takes `'Linear'`, `'Quadratic'`, `'Exponential'`; two number columns also take `'Power'` and `'Logarithmic'`), and `fit_trendline(kind, x, y)` returns the curve (`.x` ascending, `.y`). It refuses a fit the values cannot give, such as a missing value, a 0 under a log, or too few distinct X values for the kind (`PostProcessingRefusedError`, `operation='Trendline'`, `reason='fit_impossible'`, naming the values and how many).

The builds run on the session's own post-processing lane, one at a time in the order asked. A protocol run writes its images on a different lane, so a long build never delays a run's writes. The tiling config and the turret are the session's own: a caller passes neither.

### Run state and locks

The session derives all run and lock state from its activity claim.
These are the members a GUI-quality client binds its widget state to --
the same derivations LVP's own GUI mirrors into kv properties
(alongside the run predicates shown under Running protocols):

```python
session.run_lockout              # True during a run, a diagnostic, OR a run's post-run file drain
session.is_protocol_running      # True while a protocol-class run holds the scope (a run lent a
                                 # diagnostic's claim included)
session.run_in_progress          # the engine's run in any phase, its teardown included
session.held_by_other(run)       # the scope is held by anything but `run` (what start() returned);
                                 # False for the run itself, so its Stop stays live; None: held at all
session.protocol_files_draining  # run files still writing after a run finished
session.protocol_files_pending   # how many of those writes are left (0 when not draining); poll it --
                                 # the count changes between transitions, the listener fires only on them
session.protocol_files_stalled   # the drain's write in flight has stopped progressing, judged by the
                                 # same threshold that refuses a new run (files_writing_stalled)
session.protocol_files_stuck_write  # the write in flight, named for a stall report
session.exclusive_activity       # None | 'protocol' | 'recording' | 'diagnostic'
session.controls_locked          # full control-surface lock (any run lockout, or a live recording)
session.motion_enabled           # user stage motion allowed right now
session.manual_recording.is_recording  # a manual recording is LIVE (not its file drain)
session.recording_active         # a manual recording holds the scope and is live (False in its drain)
session.close_drain_pending      # video frames still queued: a recording's drain, or a run's video tail
session.close_drain_frames       # how many of those frames, across both drains (0 when none)
session.discard_close_drain()    # a closing host's escape: drop every queued frame in both; written ones stay

def on_run_state():              # called on EVERY run-state transition: an activity taking
    print(session.run_lockout)   # or releasing the scope, a manual recording going live, going
                                 # to its drain and finishing, a run's video step finishing,
                                 # a run's files all landing;
                                 # re-read the derivations (level semantics, no payload)
session.add_run_state_listener(on_run_state)
session.notify_run_state()       # force a level-sync of all listeners
```

### Outcomes

A call that fails or is refused raises to you; that exception is the call's outcome, and it is yours to report. Outcomes that no caller waits on -- a camera stream that stops, a run that ends short of its captures, a recording the disk floor stopped, a finished run's file writer that stalls, a notice that a capture was saved without its position -- reach you through the session's outcome subscription instead:

```python
def on_outcome(n):               # runs on the thread that reported it: return promptly, never wait on the scope
    print(n.kind, n.title, n.message, n.reason, n.shown, n.outcome_id)

session = ScopeSession.create(settings=settings, outcome_listener=on_outcome)   # hears bring-up too
# or, for a client that comes later:
session.add_outcome_listener(on_outcome)
session.remove_outcome_listener(on_outcome)    # shutdown() removes it too
```

Each call carries one `Notification` (`modules.notification_center`):

| Field | Meaning |
|---|---|
| `kind` | `OutcomeKind`, a string enum: `'refusal'` (declined; nothing broke), `'fault'` (something failed), `'notice'` (information; nothing failed), or `'unclassified'` (a notification posted without declaring its kind; these are being moved to declared kinds) |
| `title`, `message` | The heading and the sentence, written for the person |
| `reason` | The outcome's machine-readable code, stable enough to branch on (a refusal's, a fault's or a notice's); empty for one that declares none |
| `remedy` | A `Remedy` when the outcome has one action that answers it: `session.apply_remedy(n.remedy)` takes it |
| `solicited` | True when it answers a request a person or caller just made |
| `fatal` | True for a fault that ends what was running |
| `shown` | Whether the scope says this is for display now. False when it was muted: during a scan, a protocol, an autofocus scan of every step, or a run under a diagnostic's claim (non-fatal outcomes; a standalone autofocus, composite or z-stack is shown, whoever started it); during a standalone autofocus, composite or z-stack, a non-fatal outcome identical to one already shown in that run (same category, title and message); within 10 s of the same title being shown; or during shutdown. You receive muted outcomes too; a display shows only `shown` ones |
| `outcome_id` | One per outcome. An outcome delivered muted and later shown (because someone asked for it) arrives twice with the same id, `shown` False then True; keep the first of an id to count each outcome once |
| `wall_time`, `timestamp` | Wall-clock seconds, and a monotonic time for ordering within the process |
| `severity`, `category`, `operation_key` | The log level, the subsystem, and the operation a notice-then-outcome pair is about |

The subscription is process-wide: a listener hears every session in the process, and the mute and dedup state is shared. A listener that raises is logged at ERROR with its traceback and the others are still told. A session that already exists has finished bring-up; what bring-up reported is heard only by a listener given to `create`. What it found is held for a client that comes later:

```python
record = session.bring_up_record()         # modules.lumascope_api.bring_up.BringUpRecord
for part in record.parts:                  # 'motor', 'led', 'camera', in that order
    print(part.part, part.up, part.expected, part.cause, part.detail)
record.missing                             # the parts this scope's model has and lacks
record.part('led').cause                   # None, or why: 'not_detected', 'port_in_use', 'not_responding',
                                           # 'connect_failed', 'no_driver'; for the camera 'camera_in_use',
                                           # 'camera_port_in_use', 'camera_not_detected', 'camera_not_initialized';
                                           # on an LED board that came up, 'safety_off_failed'
record.substitution('binning')             # Substitution(setting, saved, used), or None; also 'frame'
record.settings_set_aside                  # SettingsSetAside(path, reason) while the app runs on the shipped
                                           # template because the user's file could not be used; else None
```

Bring-up reports these once, as outcomes, to the listener given to `create`: a camera that did not come up is a fault under its own heading (`reason` as above), an LED board missing on a scope whose other parts came up is `LedBoardUnavailableError` (`reason` the cause), a part the model has and lacks is listed in one `PartialHardwareError` (`'partial_hardware'`, each part with its cause), a refused connect-time LEDs-off is `LedSafetyOffNotTakenError`, and when nothing came up the one outcome is the notice `NoHardwareDetectedNotice` (`'no_hardware'`). A saved binning the camera cannot take is substituted and reported once as a notice (`'binning_substituted'`); the record holds the saved value beside the one that ran. A saved frame larger than the scope delivers at the binning applied is refitted to its maximum, reported once as the notice `FrameRefittedNotice` (`'frame_refitted'`) and recorded the same way (`substitution('frame')`), and the session stores the frame that ran, so the next bring-up has nothing to refit. The session stores the binning that ran in the saved one's place, with the frame the camera delivered (the two are one geometry). The image mode is never substituted: every camera honours every mode. A manual scope's missing motor board is expected and reported nowhere.

### Holding the scope for a diagnostic

A script that drives the hardware directly -- a characterization, a
bench measurement, anything that homes, moves, lights or grabs outside a
run -- holds the scope for its duration, so a run or a recording cannot
start in the middle of it:

```python
from modules.exceptions import DiagnosticRefusedError

try:
    with session.diagnostic_claim():   # the claim, released when the block ends, even on a raise
        ...                            # drive the hardware
except DiagnosticRefusedError as e:    # a run, a recording or another diagnostic holds the scope
    print(e.reason, e.holder, e.message)
```

While the diagnostic holds the claim it counts as holding the whole scope:
`exclusive_activity` reads `'diagnostic'`, `run_lockout` and
`controls_locked` read True, a run start is refused
(`ProtocolRunRefusedError`, `exclusive_activity_running`), a recording start
is refused (`RecordingRefusedError`, `holder='diagnostic'`), and an objective
or labware change raises `HardwareCommandRefusedError`. `is_protocol_running` stays
False: a diagnostic is not a run.

A diagnostic that needs autofocus runs the public one under its own claim,
`runner.run_autofocus(layer, claim=held)` with the `held` the block yields.
That run acts inside the diagnostic: it is not refused by it and cannot end
it, and its non-fatal outcomes are muted (`shown` False) -- the diagnostic
gets the run's outcome from its handle and decides what to show. When the block ends with such a run still live, the release waits for
the run first; if it is still live after the wait, the claim stays held and
the block's end raises `RuntimeError`.

### Support report

Two members make the ZIP Etaluma support asks for. Each blocks until the ZIP
is written and returns where it is:

```python
from modules.exceptions import SupportReportNotSavedError

try:
    saved = session.make_support_report(        # minutes: boards, motors, camera, then the files
        include_bandwidth_test=False,           # True adds a camera frame-delivery timing
        output_dir=None,                        # the Desktop; the home folder when there is none
        on_progress=lambda pct, msg: print(pct, msg),
    )
    saved = session.make_logs_zip(output_dir=None)   # seconds: logs, data folder, recent protocols,
                                                     # video receipts; touches no hardware
except SupportReportNotSavedError as e:         # no ZIP; the failure's own words, chained from it
    print(e.report, e)                          # 'support report' or 'logs zip'

saved.path      # the ZIP
saved.title     # 'Support Report Saved' or 'Logs Zip Saved'
saved.message   # the folder it is in, and the address to send it to

health = session.plugin_health()   # modules.plugins.PluginHealth, or None when no plugins load here
health.namespaces                  # each namespace's NamespaceHealth: loaded, last_runtime_errors
health.not_loaded                  # PluginNotLoaded(name, version, reason) for each that did not load

session.plugin_api_level           # int: what this host does that a plugin may rely on ("Plugin API level")
```

The full report holds the scope for a diagnostic around its hardware steps
(see above); started while a run holds the scope, it skips those steps and
says so in the ZIP. A step that fails is written into the ZIP and the report
goes on: only a ZIP that could not be written raises. Neither member is
cancellable.

Both ZIPs carry `bring_up.json` (the session's `bring_up_record()`, each
part with `cause_words` beside its `cause`) and `plugins.json` (each plugin
namespace's health, and `not_loaded`: the plugins that did not load and
why). A session gets plugin health only from a host that loads plugins and
passes `ScopeSession.create(..., plugin_health=registry.health)`; any other
writes `{"plugins": null, "why": "no plugin registry on this host"}`.

#### Plugin API level

`session.plugin_api_level` (`modules.plugins.PLUGIN_API_LEVEL`) says what
this LumaViewPro does that a plugin may rely on, within its major version.
A version cannot say it: `version.txt` names the promoted release, and many
trunk commits share one beta number. A commit that changes behaviour a
plugin relies on raises the level by one. A plugin that needs a level checks
it at its own entry and refuses below it, naming both levels; a host older
than the level has no `plugin_api_level`, which reads as 0
(`getattr(session, 'plugin_api_level', 0)`). `PluginSpec.requires_lvp_version`
still states the major version.

| Level | Added |
|---|---|
| level 1 | `capabilities.camera_analog_gain_max_db`; `capabilities.camera_reports_temperature`, and `get_camera_temperatures_degc` answering `{}` only with no sensor, `None` with no camera, and raising on a failed read; `save_camera_state` / `restore_camera_state` covering pixel format, frame size, binning and black level. |
| level 2 | `imaging.get_black_level_range()`: the range `set_black_level` accepts for the current pixel format. |
| level 3 | With no LED board: `led_on` and a lighting transition raise `not_connected`, the LED state reads (`get_led_state`, `get_led_states`, `save_led_state`) answer `None`, and `capabilities.led_max_ma` / `led_channels` are `None`; `led_on` for an LED the model lacks raises `axis_absent`. |

### Configuration queries

```python
session.get_layer_configs()              # all layer settings
session.saved_focus('Blue')              # the Z saved as a layer's focus; FocusNotSavedError when none was
                                         # ever saved (a step built for that layer takes the current Z)
session.scope.runtime_state.resolve_current_objective()  # (id, info) of the active objective; ObjectiveUnknownError when unknown
session.capture_settings_snapshot()      # settings snapshot with objective_id set to the active objective,
                                         # for composing a capture or run; not for saving
session.get_current_plate_position()     # current XY in plate coords; ConfigError when the stored plate is not in
                                         # the catalogue; HardwareCommandRefusedError('not_connected') when this
                                         # model has a motor controller and none is connected
session.get_auto_gain_settings()         # auto-gain config
session.get_stim_configs()               # stim settings per layer
session.get_enabled_stim_configs()       # only the enabled ones
```

Assembling the configuration a sequenced run takes:

```python
# Everything the run needs, from this session's settings -- no GUI involved.
config = session.get_sequenced_capture_config()

# Tiling and z-stacking are ARGUMENTS, not stored settings. Neither survives
# a restart, so there is nothing for the session to read them from: state
# what you want.
config = session.get_sequenced_capture_config(tiling='2x2', use_zstacking=True)
```

A layer whose focus was never saved (`get_layer_configs()[layer]['focus']`
is `None`) is imaged at the stage's Z when the config was built: the
config carries it as `current_z`, read from `get_current_plate_position()`.
A layer with a saved focus keeps it.

The GUI builds the same configuration through the same builder, supplying
those two from its own controls, so a scripted run and a run started from the
screen are assembled identically.

### Cleanup

```python
# Full teardown of everything the session constructed: the FILE lane, the
# worker pool and the session's threads always. On a scope the FACTORY built
# (create() with no scope=) also: LEDs off, motion stopped, scope disconnected,
# which shuts its IO and CAMERA lanes. Reading that scope afterwards:
# motor_connected is False, imaging.is_streaming() is False,
# diagnostics.get_motor_info()['model'] is None. A second shutdown() logs one
# info line and does nothing.
session.shutdown()

# For a scope YOU passed as create(scope=...): shutdown() leaves it connected
# with its lanes running, so the disconnect is yours.
session.shutdown()
session.scope.disconnect()
```

If a part does not shut down cleanly, `shutdown()` (and `scope.disconnect()`)
still runs every teardown step, then raises `ScopeDisconnectError`
(`modules.exceptions`): its `parts` name each part that failed, in teardown
order (`'motor stop'`, `'LED board'`, `'motor board'`, `'camera'`), `causes`
holds each part's error, and the first is its `__cause__`. Everything that
could be torn down has been, so a second `shutdown()` completes. Neither
returns a value.

---

## scope.motion

Axes available depend on the scope — always check `scope.capabilities.axes`.

```python
# Homing (required before movement)
scope.motion.home()                              # home everything the board has (axis='ALL' default)
scope.motion.home(axis='Z')                      # Z only
scope.motion.home(axis='T')                      # turret only (parks Z at 0, homes T, restores Z)
# Each blocks until the home is done and returns only when it established
# the reference. Unknown axis raises ValueError. A home that could not
# raises: HardwareCommandRefusedError (reason 'not_connected' or 'axis_absent':
# no motor controller, or no Z or turret to home; nothing moved) or HomingFailedError (title 'Homing Failed';
# .reason 'failed' | 'error' | 'unread', .axes the axes left unknown,
# chained from the driver's error when there is one).
scope.motion.has_homed()                         # True if the stage/focus axes know where they are
scope.motion.position_is_known('T')              # turret-specific
scope.motion.axes_without_position()             # {axis: 'unknown' | 'homing'}; {} when all known

# Homing is REQUIRED, not advisory. A commanded move on an axis whose
# position is unknown raises AxisStateUnknownError instead of driving --
# there is no reference frame for it to be absolute in. An axis is
# unknown before its first successful home, and again after a home
# fails, the board disconnects mid-move, or a move stalls out. So a
# headless or REST caller homes first and stops on the raise:
#
#   try:
#       scope.motion.home('ALL')
#   except (HomingFailedError, HardwareCommandRefusedError):
#       ...  # do not command moves; the reference frame is not established
#
# A home reads every axis's position from the board before it marks the
# axis known. When the mechanics succeeded but an axis's position could
# not be read, the home raises HomingFailedError (reason 'unread', naming
# the axis), that axis is UNKNOWN, and its last cached number is never
# re-labelled as a position. The home itself shows no popup: the caller
# that waited on it reports it.
#
# has_homed() / position_is_known(axis) answer from that same live state, so they
# report False after a fault revokes a reference that was previously
# good -- not merely "a home once succeeded".

# Position queries (µm for XYZ, 1–4 for turret). Read cache, no serial I/O.
scope.motion.get_current_position('Z')           # polled position: refreshed while the axis moves, the last read when idle
scope.motion.get_current_position()              # dict of the axes that have a position
scope.motion.axis_positions()                    # {axis: AxisPosition(state, position)} in ONE snapshot; position is None unless the axis is IDLE or MOVING -- the read for a caller writing a position into a file
scope.runtime_state.plate_transform()            # (sx_um, sy_um) -> (px_mm, py_mm), BOUND to the labware and offset registered now; None when either is unset. For a caller converting many positions over time (a recording, one per frame): every frame is stated in one frame of reference, and it cannot raise
scope.motion.get_target_position('Z')            # the commanded target, moving or arrived; the polled position when none was reached (before a first move, after a home, a STOP or a fault)
scope.motion.get_actual_position('Z')            # hardware position via serial (slow; use sparingly)
# A position with no hardware behind it is None: an axis the scope does not
# have, and every axis with no motor board installed (a manual scope, a board
# that never came up, after disconnect()). A present axis keeps answering its
# number, its reference lost or not; a pulled cable leaves the board installed.
# A position the controller does not report is never answered with a number:
# get_actual_position raises HardwareError, and HardwareCommandRefusedError
# ('not_connected') when the installed controller is not connected. A move that needs a position it cannot
# read -- a relative move's base, or Z for the backlash approach -- is refused
# with HardwareCommandRefusedError ('position_unread'); nothing is driven.

# Stop + tuning
scope.motion.stop_motion()                       # stop all in-flight moves (the app-level abort for the move_* family)
# A STOP the board did not take raises MotorStopFailedError (title 'Motor Stop
# Failed', chained from the driver's error): the stage may still be moving.
scope.motion.set_acceleration_limit(50)          # motor acceleration cap, percent of max (1-100, else AccelerationLimitRefusedError)
scope.motion.set_precision_mode('Z', True)       # per-axis precision mode on the motor board

# Absolute moves (µm). Each returns once the axis has arrived, or raises
# MoveNotCompletedError saying why it did not.
scope.motion.move_absolute('Z', 5000)
scope.motion.move_absolute('X', 60000)

# Started moves: return once the board has taken the command, for work done
# while the axis travels or axes moved together. The handle's wait() gives
# the outcome move_absolute would have given.
x = scope.motion.start_move_absolute('X', 60000)
y = scope.motion.start_move_absolute('Y', 40000)
x.wait()
y.wait()

# Relative moves (µm), waited like move_absolute; start_move_relative is the started form
scope.motion.move_relative('Z', 100)
scope.motion.start_move_relative('Z', 100).wait()

# A move that does not complete raises MoveNotCompletedError, one object with
# .axis, .reason and .title, and the axis is UNKNOWN afterwards (except
# 'stopped' and 'superseded'): 'driver_failed' (the board did not take the command; chained
# from the driver's error), 'stalled' / 'board_lost' / 'position_unread' (the
# motion monitor gave the axis up -- the last when the board said it arrived but
# never said where; a waited move raises the very object the monitor reported),
# 'timed_out' (the wait's bound ran out), 'faulted' (set UNKNOWN by something
# else during the wait), 'stopped' (stop_motion landed on it), 'superseded'
# (another move or a home took the axis before it arrived; the axis keeps
# what that command gives it). The outcome is fixed when the move ends: a
# move that arrived reads arrived however late wait() is called, whatever
# moved or stopped the axis since.

# Jog step under the active objective (z_coarse / z_fine for Z, xy_coarse / xy_fine for X, Y);
# ObjectiveUnknownError when the objective is unknown -- no step is guessed
step = scope.motion.jog_step('Z', coarse=True)
scope.motion.move_relative('Z', -step)

# Status
scope.motion.get_target_status('Z')              # True if target reached
scope.motion.is_moving()                         # any axis moving?
scope.motion.wait_until_finished_moving()        # block until the axes moving now stop; raises MoveNotCompletedError
                                                 # if one ended UNKNOWN, or 'still_moving' if the wait ran out
scope.motion.position_is_known('Z')              # False until homed: an absolute move would refuse

# Limit switches -- why a move stopped short. Reaching a limit is reported,
# not raised, so a move that ran out of travel and one that arrived look the
# same until you ask.
scope.motion.get_limit_switch_status('X')        # (left, right); 1 engaged, 0 clear, -1 unreadable
scope.motion.get_limit_switch_status_all_axes()  # dict of axis -> that pair, for the axes the board has

# Turret
scope.capabilities.has_turret                    # turret presence probe
scope.motion.move_turret(2)                      # turret position 1-4; any other slot raises
                                                 # PositionOutOfRangeError rather than driving
scope.motion.get_turret_slot()                   # slot in the light path, or None when not known
# The turret has no encoder: the slot is the one the last move_turret or home
# left it in, recorded only when that command returned without error and no
# stop_motion landed on it. None before the first, while one runs, after one
# fails (MoveNotCompletedError), and after the turret's position is lost.
# move_absolute / move_relative refuse 'T' with
# ValueError -- the turret moves only by slot, through move_turret.
scope.motion.get_preferred_turret_slot()         # the slot the last move_turret landed on, or None
# Never written by a home, and saved as turret_position, so it survives a
# restart. When two slots carry the same objective, the slot lookup
# (get_turret_position_for_objective_id) prefers it, then the current slot,
# then the lowest-numbered -- a run and step navigation choose alike.

# Stage
scope.motion.get_axis_limits('Z')                # {'min': 0, 'max': 14000}, read-only
```

The limits are a read-only mapping: they are the bound a move is refused
against, so an edit raises `TypeError`. Take `dict(...)` of it for a
working copy.

**Axes: ask which axes the scope has before asking their limits.**
`scope.capabilities.axes` names the axes this scope has a motor for.
`get_axis_limits(axis)` answers the motion board's travel for an axis,
and the board answers X and Y on a Z-only scope too, so read limits only
for the axes `capabilities.axes` names. The turret's `'T'` answers None:
it moves by slot, not by distance.

**Z overshoot:** firmware moves below target then approaches from below, eliminating leadscrew backlash for consistent focus. The leg below is part of the move: Z reads `'moving'` from the first target write, `wait_until_finished_moving()` waits through the leg and the approach, and a frame grabbed during either is not valid.

**Axis state model:**

```python
from modules.lumascope_api import AxisState

scope.motion.get_axis_state('Z')          # 'idle', 'moving', 'homing', or 'unknown'
scope.motion.is_any_axis_moving()
```

**Position listeners** (push-based):

```python
def on_position(axis: str, target: float, state: str):
    print(f"{axis} → {target:.1f}µm ({state})")

scope.motion.add_position_listener(on_position)
scope.motion.remove_position_listener(on_position)
```

---

## scope.illumination

Channels available depend on the scope — always check `scope.capabilities.led_colors`.

**A channel is named, never numbered.** `'BF'`, `'PC'`, `'DF'`, `'Blue'`,
`'Green'`, `'Red'` — the name is the portable identity. The number behind it is
a driver detail and is NOT portable: an FX2 board carries four channels and an
RP2040 board six, so the same integer means different LEDs, or none, depending
on the board. Ask `caps.led_colors` for what a given scope can actually drive.
Passing an integer still works today as a legacy compatibility form, but it is
not the supported contract and will not be part of the REST surface.

**LEDs the scope cannot drive**: `led_on` asks, in order, whether the channel is one at all, whether an LED controller is connected, and whether this model has the LED. A name that is no layer raises `ConfigError`, and a number outside the installed board's table `ValueError`. With no LED controller connected it raises `HardwareCommandRefusedError` (`not_connected`, `MissingPart.LED_CONTROLLER`). An LED this model does not have -- a layer it lacks (an LS560's `'Red'`), a layer that drives no LED (`'Lumi'`), or a number no layer of the model drives (an LS560's `2`) -- raises `HardwareCommandRefusedError` (`axis_absent`, `MissingPart.led(...)`), and nothing lights; it never maps to a substitute channel. A model this release does not know (its layer identity unresolved) takes the board's numbers. `led_off` of an LED the scope does not have is an idempotent no-op — a channel the scope does not have is already off.

**Luminescence** (`Lumi`): not an LED channel. In luminescence mode, all LEDs must be off — the image captures emitted light only.

```python
scope.illumination.led_on('Blue', 200)                 # Blue LED at 200 mA
scope.illumination.led_on('Blue', 200, block=True)     # wait for firmware confirmation
scope.illumination.led_off('Blue')
scope.illumination.leds_off()                          # turn off all LEDs

# Channel mapping. Numbers are a DRIVER detail -- these exist to read the
# board's own wire vocabulary, not to address channels from L2.
scope.illumination.color2ch('Blue')                    # 0  (or None if the scope doesn't have this color)
scope.illumination.ch2color(0)                         # 'Blue'

```

**Safety limits** (enforced by firmware on RP2040 boards): per-channel max 1000 mA, board total max 3000 mA. FX2 boards have their own per-channel cap declared in the camera profile.

**The current is the board's step, not your number.** Each board drives its LEDs in steps: whole mA on the EL-0940 LED board, and about 3.29 mA (one step of 840 mA over 255) on the FX2 scopes (LS560, LS620, LS720). `led_on` lights the channel at the nearest step, a half going up; a request above zero but below one step lights at one step, never dark; 0 mA leaves the channel on at 0 mA, which the board receives as off. `led_on` returns that commanded current, and it is what `get_led_state`, the LED listeners, a frame's record and a saved file's `Illumination` report: `led_on('BF', 1.0)` on an LS620 returns about 3.29, and the file says so. It is the current the board was commanded, not a measurement. The layer setting and a protocol step's `Illumination` keep the value you wrote, so a protocol stays the same when it moves between scopes; `api.log` names both whenever they differ.

### State queries — read from the API, never the driver

Lumascope holds the authoritative LED state in an internal cache. The API layer's `get_led_state()` / `get_led_states()` read from that cache. **Never call the driver's state methods directly** — for FX2 scopes the driver is a pure command translator and its state queries return sentinels.

```python
scope.illumination.get_led_state('Blue')               # {'enabled': True, 'illumination_ma': 200} when on; {'enabled': False, 'illumination_ma': None} when off
scope.illumination.get_led_states()                    # all channels, same per-channel shape as get_led_state
scope.illumination.save_led_state('mine')              # a snapshot for restore_led_state
```

With no LED board installed -- none came up, or after `scope.disconnect()` -- the three answer `None`: there is no board whose state to describe, which is not the same answer as every channel off. `disconnect()` darkens the board and leaves nothing believed lit. A frame captured then records no LED lit.

### Exclusivity — a run holds the LEDs

LED writes carry no owner name: no string grants the right to drive an LED.

```python
scope.illumination.led_on('BF', 200)
scope.illumination.led_off('BF')
scope.illumination.leds_off()                          # unconditional off (shutdown / cleanup)
```

**A run can hold the LEDs exclusively, and a write refused on that account is SILENT.** While a protocol run, autofocus or another subsystem holds the internal LED lease, an `led_on` / `led_off` from anyone else is refused: the LED does not change, the refusal is recorded in `api.log`, and **the call raises nothing.** A refused `led_on` returns `None` where one that commanded returns the current; a refused `led_off` returns `None` exactly as a successful one does. The refusal is deliberate — it stops a live UI change from disturbing a run's channels — but nothing is raised, so unless your code checks `led_on`'s return a capture taken afterwards can come back dark with nothing in your own code to explain why. If an LED command appears to do nothing, check whether a run is in flight before suspecting the hardware. (Making this refusal visible at the API boundary is open work; the lease itself is internal machinery and not L2 surface.)

### Save / restore — the autofocus pattern

Preserve the user's LED state while a subsystem does its own work, then restore:

```python
# User has Red on at 150 mA. Autofocus needs BF:
snapshot = scope.illumination.save_led_state('autofocus')        # capture current state
scope.illumination.led_on('BF', 100)
# ... autofocus runs: changes Z, captures frames, evaluates focus ...
scope.illumination.restore_led_state(snapshot)                   # Red back on at 150 mA, BF off
```

`save_led_state(tag)` returns a snapshot dict (the tag is for logs); `restore_led_state(snapshot)` turns off lit channels the snapshot does not have on and re-lights the ones it does. While a run holds the LEDs, the restore is refused like any other write.

### Listeners — push-based notifications

Prefer listeners over polling. Listeners fire on every LED state change (enable, disable, illumination change) with no serial I/O cost:

```python
def on_led(channel: str, enabled: bool, illumination_ma: float):
    print(f"{channel} {'ON' if enabled else 'OFF'} {illumination_ma}mA")

scope.illumination.add_led_listener(on_led)
# ... later ...
scope.illumination.remove_led_listener(on_led)
```

Use polling only when you specifically need the current value at a moment in time (e.g., settling a UI field to match hardware after a reconnect). For "did anything change?" questions, always use listeners.

---

## scope.imaging

Camera capture and configuration live on the `scope.imaging` sub-API
namespace. The methods below are the L2-stable surface; the underlying
driver is `scope.imaging._driver` (private; reach through the API).

```python
# Streaming control. connect() returns the camera CONFIGURED but NOT
# grabbing; capture/get_image need a live feed, so start it first. The
# Session factories release the gate at bring-up, after initialize; a bare
# Lumascope you constructed yourself needs the explicit call.
scope.imaging.start_streaming()   # begin the live feed (idempotent; also
                                  # restarts a feed stopped via stop_streaming)
scope.imaging.stop_streaming()    # stop the feed (get_image then times out)
scope.imaging.is_streaming()      # True while acquiring (queries the driver)
```

```python
# Raw frame grab (no validity wait — use capture_and_wait instead in most cases)
image = scope.imaging.get_image()
image = scope.imaging.get_image(force_to_8bit=False)   # keep native 12/16-bit
# Returns numpy.ndarray on success, None on failure (camera inactive,
# frame drain failed, timeout). Per the sentinel-return contract:
#   if image is None: ...
#
# Shape is (H, W) 2D mono for mono-native cameras and (H, W, 3) RGB
# for color-native cameras (see scope.capabilities.is_color_native).
# Layer false-color is NOT applied here -- apply at the display /
# encode boundary via image_utils.mono_to_rgb_falsecolor(img, layer).
# Dtype is uint8 with force_to_8bit=True (default) or for 8-bit
# cameras; uint16 with force_to_8bit=False for 12/16-bit cameras
# (see scope.capabilities.native_bit_depth).
#
# Payload depth of a frame you just captured (for scaling / saving a
# uint16 frame): scope.imaging.last_significant_bits -- the per-frame
# delivery stamp (e.g. 12 for Mono12 in a uint16 container). Prefer it
# over scope.imaging.significant_bits (derived from the current pixel
# format) when you are holding the frame -- the stamp cannot describe a
# newer format than the frame was captured under.

# Frame-validity capture — PREFERRED for all real captures.
# Waits for all pending changes (LED, gain, exposure, motion) to settle,
# drains stale frames, returns a valid frame. Returns None on failure.
# The capture honors invalidation across its WHOLE window: a state change
# landing after the drain — during the grab itself — is detected, and the
# capture re-drains, re-derives its expectations, and grabs again, so the
# returned frame always reflects the state you last commanded. A capture
# contended by state changes is therefore slower than an uncontended one.
# The settle-and-recheck work is bounded by a deadline sized from the
# pending work at entry (frames to settle, exposure, sum window): if
# invalidation keeps arriving faster than frames can settle it, the
# capture returns None in bounded seconds with "DEADLINE EXPIRED" named
# in the log, instead of holding indefinitely. The deadline suspends
# while commanded stage motion is still physically settling — a capture
# issued during a long move waits for the move, as it should.
# The dark-floor expectation is DERIVED from commanded LED state: a
# channel counts as lit only at strictly positive current, so a channel
# commanded at 0 mA is dark by design, as are luminescence captures and
# any capture with nothing commanded. With a channel lit, a frame with
# essentially no lit pixel is retried until timeout_s -- which heals a
# stale pre-LED frame when a lit one is on its way -- and then RETURNED,
# carrying 'dark_saved': True on last_capture_info. It is never refused:
# pixel content cannot distinguish an LED that failed from a genuinely
# dark sample, so darkness is reported, not acted on. accept_dark=True
# (keyword-only, default False) skips the measurement entirely for the
# callers whose dark frames are expected: custom focus sweeps (an
# out-of-focus fluorescence plane can carry no signal) and benchmark
# probes; those captures file no dark_saved fact.
image = scope.imaging.capture_and_wait()
image = scope.imaging.capture_and_wait(
    force_to_8bit=True,
    accept_dark=False,                     # True skips the darkness measurement
    all_ones_check=True,                   # detect saturated frames
    sum_count=4,                           # SUM 4 frames (not an average); a
                                           # sum is uint16 on every camera and
                                           # saturates at 65535
    sum_delay_s=0.05,                      # delay between sum frames
    exclude_sources=(),                    # sources not to wait for; the frame can then
                                           # predate that change (see Frame validity)
)

# Exposure (milliseconds) + gain (dB)
scope.imaging.set_exposure_ms(exposure_ms=50)
scope.imaging.get_exposure_ms()                  # last-known-good on transient read failure; 0.0 camera-absent
scope.imaging.set_gain_db(gain_db=10.0)
scope.imaging.get_gain_db()                           # last-known-good on transient read failure; -1.0 camera-absent

# Live-confirmed readings for metadata / records: only fields whose
# driver read succeeded RIGHT NOW; a field whose read failed is omitted
# (the value getters above would answer last-known-good instead).
scope.imaging.get_live_camera_settings()           # any of: gain_db, exposure_ms,
                                                   #   frame_size {'width','height'}, pixel_format;
                                                   #   {} when no camera is active

# `set_exposure_ms` warns + logs a stack trace at < 0.005 ms (the
# common L1 failure is typing 0.05 thinking microseconds and getting
# a black image).
#
# Bench and characterization scripts that sweep deliberately extreme
# values can silence that warning for a block -- and ONLY for a block,
# so a sweep does not disable the warning for the rest of the process:
#
#   with scope.imaging.suppress_value_warnings():
#       for ms in (0.001, 0.002, 0.004):
#           scope.imaging.set_exposure_ms(ms)

# Every camera-settings setter in this section dispatches to the camera
# lane and BLOCKS until applied (returns the body's own result). While a
# protocol run or recording owns the hardware, these raise
# HardwareCommandRefusedError instead of interleaving with the run --
# same refusal contract as the motion and LED commands.

# Batched settings (gain + exposure + auto-gain in one call)
scope.imaging.apply_layer_camera_settings(
    gain_db=5.0, exposure_ms=50,
    auto_gain=False, auto_gain_settings=None,
    layer='BF',          # names the layer in api.log; optional, defaults to '(unspecified)'
)

# Auto-exposure: the camera's own exposure control, where the body has one.
# Blocks until the camera has applied it. Check
# caps.camera_supports_auto_exposure first -- not every body offers it.
scope.imaging.set_auto_exposure_time(True)

# Auto-gain: the continuous toggle, the one-shot settle, and the setpoint
scope.imaging.set_auto_gain(True, settings={'target_brightness': 0.3, 'min_gain_db': 0.0, 'max_gain_db': 20.0})
scope.imaging.auto_gain_once(True, target_brightness=0.3, min_gain_db=0.0, max_gain_db=20.0)
scope.imaging.update_auto_gain_target_brightness(0.5)   # live setpoint tweak while auto-gain runs

# A camera without the mode (caps.camera_supports_auto_gain /
# _auto_exposure False: the IDS U3-34L family, the LS620's FX2) is never asked.
# Turning the mode on, or setting its target, raises
# CameraSettingUnsupportedError (reason 'auto_gain_unsupported',
# 'auto_exposure_unsupported' or 'auto_gain_target_brightness_unsupported');
# turning it off is that camera's state already, so it returns without a write.
# apply_layer_camera_settings with auto_gain=True applies it as manual there --
# see applied_auto_gain_for below.

# Camera-model-specific tuning knobs. Probe support first:
# scope.capabilities.camera_supports_conversion_gain_mode / _line_noise_reduction.
scope.imaging.set_conversion_gain_mode('High')     # True when applied; False when unsupported / no camera
scope.imaging.set_line_noise_reduction(True)       # same contract

# Black level: the camera's own offset parameter, in the camera's own units
# (Basler: a model-specific step, 0.0625 DN at 12 bits on the daA3840; IDS:
# DN of the current pixel format; FX2: the sensor's Row Black Target, fixed).
# Recorded with every frame (FrameRecord.black_level) and in saved metadata;
# a failed read at capture raises HardwareError and fails the capture.
scope.imaging.get_black_level()                    # live read; None when no camera or none reported;
                                                   #   raises HardwareError when the read fails (no cache)
scope.imaging.get_black_level_range()              # (minimum, maximum) set_black_level accepts, live, for the
                                                   #   current pixel format; None when no camera or none settable;
                                                   #   raises HardwareError when the read fails
scope.imaging.set_black_level(value=4.0)           # value in effect; probe
                                                   #   scope.capabilities.camera_supports_black_level first;
                                                   #   raises HardwareError when the camera write fails

# Frame rate: the camera's own figure for what its current settings allow
# (Basler: the resulting acquisition frame rate; IDS: AcquisitionFrameRate's
# maximum; the simulator: its pacing). The rate frames actually reach the
# host at is measured from their arrivals, not read here.
scope.imaging.get_resulting_frame_rate()           # fps; None when no camera or none reported (FX2);
                                                   #   raises HardwareError when the read fails (no cache)

# Frame size (getters answer last-known-good on a transient read
# failure; None / 0 only when no camera is active or never read)
delivered = scope.imaging.set_frame_size(2048, 2048)
# Returns the DELIVERED {'width','height'}: the size asked for, on every
# camera (it acquires the next window up on its grid and crops back).
# Only a size within one grid step of the camera's own maximum comes
# back smaller: no window on its grid holds it.
# Raises CameraSettingRejected (modules.exceptions) when a live camera
# refuses the apply; returns None (no-op) when no camera is active.
# Base geometry code on the returned dict, never on the request.
scope.imaging.frame_size_cached                    # {'width': ..., 'height': ...} -- cache read, no driver I/O
scope.capabilities.camera_max_frame_size           # (width, height) the scope's unbinned frame ceiling: the sensor,
                                                   # or the model's smaller one (LS560: 1700); None if unknown
scope.imaging.get_pixel_alignment()                # {'width','height'} deliverable frame-size granularity: {2, 2}, even sides (H.264), on every camera

# Binning
scope.imaging.set_binning_size(2)
# True when applied; raises CameraSettingRejected when a live camera
# refuses; False (no-op) only when no camera is active. Same contract
# for set_pixel_format. Success is observed by the return value,
# rejection by the typed raise -- a dropped return cannot silently
# record a rejected apply.
scope.imaging.get_binning_size()                   # always >= 1 (last-known-good on failed read)
scope.capabilities.camera_binning_sizes            # e.g. (1, 2, 4)
scope.imaging.set_pixel_format('Mono12')           # True when applied; raises CameraSettingRejected on refusal
scope.capabilities.camera_pixel_formats            # e.g. ('Mono8', 'Mono12') -- the enumeration for set_pixel_format

# Geometry value getters (last-known-good on a transient failed read; 0 camera-absent)
scope.imaging.get_width()
scope.imaging.get_height()

# Cache reads, no driver I/O (the *_cached family)
scope.imaging.gain_db_cached
scope.imaging.exposure_ms_cached
scope.imaging.pixel_format_cached
scope.imaging.min_frame_size_cached                # dict, or None when no camera is connected

# The longest exposure the camera may be using now (ms), no driver I/O: the
# armed auto-gain's exposure ceiling while continuous auto-gain is on (the
# sensor maximum when it was given none), else exposure_ms_cached. Bound a
# wait for the next frame by this, not by the cache, which holds the value
# from before auto-gain was armed. None when neither is known.
scope.imaging.longest_exposure_ms

# Scale bar overlay (burned into frames the imaging paths return when enabled;
# skipped while the objective is unknown, with one warning each time it becomes unknown)
# Whether it is drawn is the session's setting (session.set_scale_bar); the
# colour is the imaging API's own.
scope.imaging.set_scale_bar_color('Red')
scope.imaging.scale_bar_config                     # {'enabled': bool, 'color': str | None}
```

The acquisition frame-rate cap lives on the camera driver and clamps frame production regardless of sensor-readout capability. It is a driver-level control with no public API member -- described here only to explain the behavior. Used by the manual-record path to match user-requested video FPS, and by characterization tools to bound capture rate during long-running probes. No-op on drivers that do not implement the underlying setter (warning logged). Distinct from `set_exposure_ms` (per-frame integration time) and from any host-side throttling.

### Dynamic camera capabilities

Cameras advertise their real limits at connect time. Use these to size UI sliders and clamp auto-exposure / auto-gain:

```python
scope.imaging.max_exposure_ms_cached                  # ms, None if no camera connected
scope.imaging.max_gain_db_cached                      # dB, None if no camera connected
scope.imaging.min_exposure_ms_cached                  # ms, None if the camera declares no floor
scope.imaging.min_gain_db_cached                      # dB, None if the camera declares no floor
```

A floor is `None` when the camera declares none, not `0`: a missing floor is not a floor of zero, and nothing checks against it. A run whose steps ask for a gain or exposure outside these limits is refused before it starts (reason `camera_setting_out_of_range`), naming each step; the stored values are left as they are.

These are derived from the camera's profile, which is populated at connect via `_query_dynamic_capabilities()` — live SDK queries for Pylon / IDS, hardcoded-from-datasheet for FX2. Per-camera values observed in practice: LS620 FX2 = 42.144 dB gain (128x) / 1000 ms exposure cap; Pylon/IDS ranges are driver-reported.

### What a stored setting applies as

A saved per-channel gain or exposure is the user's committed intent and outlives whichever camera is attached. A body that cannot reach the value is driven to its own maximum instead, and the stored value is left alone — reconnect a capable camera and the intent applies again. Ask the API what a stored value actually becomes rather than comparing against a cap yourself:

```python
applied = scope.imaging.applied_gain_db_for(48.0)     # stored=48.0 applied=20.0 capped=True
applied = scope.imaging.applied_exposure_ms_for(500.0)
applied.stored                                        # what the user set
applied.applied                                       # what this camera is given
applied.capped                                        # True when the body is holding it down
```

The same holds for a stored auto-gain preference. A camera without hardware auto-gain runs manual whatever the layer stored, and the preference is kept for a camera that has the mode:

```python
applied = scope.imaging.applied_auto_gain_for(True)   # IDS: stored=True applied=False capped=True
```

All three return an `AppliedCameraSetting`. For gain and exposure, an unknown cap (no camera, or a driver that publishes none) narrows nothing, so `applied == stored` and `capped` is `False`. This is the only place the cap is applied: the per-layer apply path sends `applied` to the driver, so the value a caller reads back is what the sensor is actually at.

Note the difference from `set_gain_db`, `set_exposure_ms` and `set_frame_size`, which are **explicit requests** and raise `CameraSettingOutOfRangeError` for a value outside the camera's range, the same on every camera. Capping belongs to re-applying something already stored; a direct request for an out-of-range value is an error, not something to silently narrow.

### Save / restore camera state

```python
snapshot = scope.imaging.save_camera_state('autofocus')
# ... change gain/exposure ...
scope.imaging.restore_camera_state(snapshot)
```

Symmetric to the LED version, and like `restore_led_state`, `restore_camera_state` takes only the snapshot.

The snapshot is **omit-if-unknown**: it always carries `tag`, and carries `gain_db` / `exposure_ms` only when a usable value existed at save time (a missing field means that value was never successfully read from the camera; `save_camera_state` logs a warning when it omits one). Use `.get(...)` rather than indexing if you read snapshot fields directly. `restore_camera_state` restores the fields present, quietly skips absent ones (callers may deliberately trim fields they want left at current values), and leaves the camera unchanged for anything it skips.

The snapshot also carries `auto_gain_arm`: the standing continuous auto-gain arm at save time, or `None`. `restore_camera_state` puts the loop back the way the snapshot found it -- re-armed when an arm was recorded (clamping the exposure to the channel class's ceiling like any arm), disarmed when none was recorded and one stands now, untouched when the field is absent. A save/restore pair around your own camera work therefore hands the live view back adjusting if it was adjusting before.

With a camera active the snapshot also carries `frame_size` (`{'width', 'height'}`) and `pixel_format`, `binning` where `capabilities.camera_binning_sizes` offers more than one size, and `black_level` where `capabilities.camera_supports_black_level` is True. These are read live, and a read that fails raises `HardwareError` from `save_camera_state`: they are never left out silently. `restore_camera_state` writes each of them only where it differs from the camera's value now (a frame size, format or binning write restarts grabbing), in the order binning, frame size, pixel format, black level, then gain, exposure and the arm; a refusal of binning, frame size or format raises `CameraSettingRejected` there and the rest is not restored. The conversion-gain and line-noise modes are not in the snapshot (they have no getters).

### Camera listeners

```python
def on_camera(param: str, value: float):
    print(f"Camera {param} = {value}")

scope.imaging.add_camera_listener(on_camera)       # fires on set_gain_db / set_exposure
scope.imaging.remove_camera_listener(on_camera)
```

### Live frame listeners

Sync per-frame handlers fire on every successful camera grab (Pylon `PylonImageGrab` thread / IDS grab loop / simulated pump). This is the canonical entry point for live image-processing plugins (see `ctx.plugins.live_processing`) and the manual-record path.

```python
def on_frame(image, timestamp, chunks):
    # Runs on the SDK callback thread. MUST NOT block. Heavy work
    # belongs on an executor.
    queue_write(image)

scope.imaging.add_frame_listener(on_frame, name='my_recorder')
# ...
scope.imaging.remove_frame_listener(on_frame)
```

- **Don't-mutate contract.** The `image` array is shared across all listeners. Write to your own output buffer if you need to keep results; mutating the array affects later listeners + downstream display / capture consumers.
- **Budget.** Each handler must complete within ~24 ms (anchored to a 30 fps target, half the inter-frame window). Over-budget invocations log a WARNING. After 30 consecutive over-budget hits, the handler is auto-removed and reported once as a `FrameHandlerRemovedError` warning (reason `over_budget`).
- **A handler that raises.** The first error of a run of failures is logged with its traceback; the rest are counted. A handler that raises on 30 consecutive frames is removed the same way (reason `raised`). A frame it handles without raising resets the count.
- **A registration the camera refuses** raises `FrameListenerNotRegisteredError` (a `CaptureError`, `reason='frame_listener_refused'`), chained from the driver's error; nothing is left registered, so the call can be retried.
- **Removal.** Once `remove_frame_listener` returns, no new call reaches the handler (a call already running completes), even if the camera driver fails to unregister it.
- **Re-entrancy.** A handler will not be re-entered on the same thread; the driver's fire-site is single-threaded.
- **Plugin authors**: use `ctx.plugins.live_processing.register(spec, handler)` rather than calling `add_frame_listener` directly. The registry forwards through to this API and surfaces the plugin name in the budget-violation log.

### Listener callback signatures (overview)

The six listener families each pass a different callback signature -- register a callable matching the row for the listener you subscribe to:

| Listener | Register via | Callback signature |
|---|---|---|
| Motion / position | `scope.motion.add_position_listener` | `on_position(axis: str, target: float, state: str)` |
| LED / illumination | `scope.illumination.add_led_listener` | `on_led(channel: str, enabled: bool, illumination_ma: float)` |
| Camera params | `scope.imaging.add_camera_listener` | `on_camera(param: str, value: float)` |
| Live frame | `scope.imaging.add_frame_listener` | `on_frame(image, timestamp, chunks)` |
| Run state | `session.add_run_state_listener` | `on_run_state()` -- no payload; re-read the session derivations (see Run state and locks above) |
| Outcomes | `ScopeSession.create(outcome_listener=...)` or `session.add_outcome_listener` | `on_outcome(n)` -- one `Notification` (see Outcomes above) |

The four `scope.*` listeners each have a matching `remove_*_listener(callback)`. The frame listener additionally takes a `name=` kwarg and carries the don't-mutate + 24 ms budget contract documented above; the other three are lightweight state-change notifications.

The last two rows are not registered on the scope. **Run state** is registered on the **session**: it takes no payload and is level-synced -- registering calls it once immediately, so a subscriber never misses a transition that happened before it subscribed, and it has no remover. **Outcomes** are registered on the session, or given to the factory to hear bring-up, and have a matching `session.remove_outcome_listener(callback)`.

### Camera info

```python
scope.camera_connected                             # bool property (mirror of motor_connected / led_connected)
scope.imaging.active_cached                        # True if grabbing
scope.diagnostics.get_camera_temperatures_degc()        # temperature sensors; {} no sensor, None no camera, a failed read raises
scope.capabilities.camera_model                    # the camera's model, serial number (camera_serial_number) and
                                                   # timestamp clock (camera_timestamp_tick_hz), read at connect
scope.diagnostics.get_camera_profile_info()        # sensor specs + dynamic ranges; None when no camera is
                                                   # connected; a failed read of a connected camera raises. Returns:
# {
#   'sensor': 'Aptina MT9P031',
#   'pixel_size_um': 2.2, 'shutter': 'rolling',
#   'gain_min_db': 0.0, 'gain_max_db': 42.144,
#   'max_exposure_ms': 1000.0,
# }
```

### Frame validity

Frame validity is the single source of truth for "is the next frame still what I asked for?" Every hardware state change invalidates pending frames. `capture_and_wait()` drains stale frames until all sources settle, detects any invalidation that lands mid-grab (re-draining and re-grabbing so the result reflects the newest commanded state), bounds the whole settle-and-recheck loop with a deadline (loud None when invalidation outruns it), and verifies the returned frame's own chunk metadata (exposure / gain) against the requested values on cameras with chunk support -- the saved frame proves its own settings.

When continuous auto-gain is armed, `capture_and_wait()` first locks it: the camera's auto loop is switched off, the exposure and gain it landed on become the requested values, and the frame is then verified against them like any manual setting. The outcome is on `scope.imaging.last_capture_info` after the call: `'auto_gain'` is `'CONVERGED'`, `'MAXED'` (exposure pinned at the layer class's ceiling, the scene too dark for the range), `'AT_MINIMUM'` (exposure at or below the class's usable floor, the scene too bright) or `'FAILED'` (the camera reported no usable achieved value, so the frame was captured without an exposure/gain check); `'auto_gain_exposure_ms'` and `'auto_gain_gain_db'` carry the locked values. The limit states are outcomes, not failures: the frame is returned and saved with its achieved values. A live-view arm is re-armed after the capture; a protocol step's arm stays locked until the next step arms again.

A caller that switches auto-gain off and wants to keep what it achieved calls `scope.imaging.lock_auto_gain()`, which returns the same result object: `state` (None when no arm was standing), `exposure_ms` and `gain_db` (the achieved values), and `stored_exposure_ms`, the exposure to store as the manual setting -- the achieved value floored to the channel class's usable floor, decided by the API so every client stores the same value. Limit states and a failed lock under a live-view arm also reach the user through the notification center; a protocol step's arm is logged only.

```python
# Leave auto-gain and keep what it achieved as the manual setting
lock = scope.imaging.lock_auto_gain()
if lock.state is not None:
    scope.imaging.set_gain_db(lock.gain_db)
    scope.imaging.set_exposure_ms(lock.stored_exposure_ms)
    print(f"{lock.state.value}: achieved {lock.exposure_ms} ms, stored {lock.stored_exposure_ms} ms")
```

```python
image = scope.imaging.capture_and_wait()
info = scope.imaging.last_capture_info or {}
if info.get('auto_gain') == 'MAXED':
    print(f"scene too dark for the range: exposure pinned at {info['auto_gain_exposure_ms']} ms")
```

```python
scope.imaging.frame_is_valid                       # True if next frame is valid
scope.imaging.frames_until_valid()                 # 0 = ready, >0 = keep draining
# To record a frame you grabbed yourself, call the frame_validity instance
# directly (see below) -- capture_and_wait handles it internally.
```

`capture_and_wait()` also rejects a frame that is essentially all white, and the measurement behind that check is available to callers who need to judge an exposure themselves:

```python
# Fraction of pixels at or above 99% of full scale, 0.0-1.0.
# significant_bits is the frame's PAYLOAD depth, not its container width --
# a 12-bit frame in a uint16 array tops out at 4095, so measuring it
# against 65535 reports a fully blown frame as 0% saturated.
image = scope.imaging.capture_and_wait(force_to_8bit=False)
frac = scope.imaging.saturated_fraction(image, scope.imaging.last_significant_bits)
if frac > 0.01:
    print(f"{frac:.1%} of pixels are clipped -- lower the exposure or the illumination")
```

For a sum, measure against `capture_frame_full_scale(image)`, the value N blown frames reach, rather than a bit depth: `capture_and_wait(all_ones_check=True)` already checks each summed frame against its own depth before summing.

A sum carries the bits it can reach: N frames of b bits reach N x (2^b - 1), so four 12-bit frames are a 14-bit sum, up to 16, where its 16-bit container saturates. Summing is for a brighter image, so every 8-bit rendering of a sum -- a JPG, LumaViewPro's display and overlays -- is drawn against one frame's white: four frames look four times brighter, and white wherever the sum passes one frame's full scale. A full-depth file keeps every count.

A whole-frame mean cannot answer this question: an evenly lit field at 70% of full scale and a field that is 70% blown white and 30% black report the same mean. Any caller deciding whether an operating point is usable needs the pixel count.

For deeper introspection (diagnostic tooling, plugin authors writing custom capture loops, advanced timing analysis), the underlying `FrameValidity` instance is available as `scope.imaging.frame_validity` and is part of the L2-stable surface:

```python
fv = scope.imaging.frame_validity

fv.is_valid                                # bool property -- next frame valid right now?
fv.is_valid_for(exclude_sources=('z_move',))  # bool -- valid if you don't care about Z motion
fv.frames_until_valid()                    # int -- drains remaining
fv.frames_until_valid(exclude_sources=('z_move',))
fv.pending_sources                         # dict {source: frames still needed} (snapshot); 0 means
                                           # the frame count is met, NOT that the source settled --
                                           # a motion source holds at 0 while its axis still moves
fv.invalidation_counts                     # dict {source: total invalidate() calls} — monotone
                                           # history frames can never erase; snapshot before a
                                           # grab and compare (!=) after to detect a mid-window
                                           # invalidation even when frames already settled it
fv.invalidate('led')                       # mark a source dirty (usually called by API setters)
fv.count_frame(frame_seq)                  # mark a frame as drained (the API's capture paths do this)
                                           # frame_seq is the arrival ordinal the driver's grab returns
                                           # WITH the frame: the same buffered frame polled twice counts
                                           # once, and a frame grabbed before a hardware write cannot
                                           # retire that write's wait. Chunk metadata never clears a
                                           # source; capture_and_wait uses it to reject a frame whose
                                           # exposure / gain disagree with what was requested
```

`set_settle_check(fn)` is the API-only registration hook for motion-completion gating and is not used by L2 callers directly. Everything else is fair game for plugin / SDK consumers.

Invalidation is automatic for normal flows — you don't need to call `invalidate()` yourself unless you're writing a custom hardware setter outside the API. The sources that invalidate frames are:

```
led        — LED turn on/off or illumination change
gain       — gain change
exposure   — exposure change
black_level — black level change
auto_gain  — continuous auto-gain armed (settles against the lit scene)
pixel_format, frame_size, binning — geometry or format change
conversion_gain_mode, line_noise_reduction — sensor noise-mode change
z_move     — Z axis motion
xy_move    — X or Y axis motion
turret     — turret move
```

Each source has its own skip count in `FrameValidity.SKIP_FRAMES`; `invalidate()` raises `ValueError` for a source with no count rather than settling it on a count nobody chose.

`exclude_sources` in `capture_and_wait()` skips the wait for a source, so the frame returned can predate that source's change -- a frame already on its way when the change went out. That fits a frame you only show; never pass it for a frame you measure or record. Autofocus waits out each Z move. Earlier builds, the 4.0.0 beta included, excluded `z_move`, so a step could score the previous step's frame at its own Z. Each sweep step now costs the `z_move` count (2 frames) after its move.

---

## scope.diagnostics

Hardware diagnostic probes and identity getters live on the `scope.diagnostics` sub-API. Per-call (no persistent state); meant for tech-support reports, bench tooling, and bring-up scripts that want one-shot snapshots of camera / motor / LED state.

```python
scope.diagnostics.get_motor_info()         # the model the board reports ('LS850'), serial, firmware, axis config
scope.capabilities.model                   # the model the scope runs as: the board's, else the selection
scope.diagnostics.get_led_info()           # firmware_version, connected, command_set:
                                           #   'v2' (INFO, SELFTEST, I2CSCAN, LEDREAD),
                                           #   'legacy' (INFO only: firmware before v2),
                                           #   None (no text commands, e.g. an FX2 scope)
scope.diagnostics.get_system_info()        # combined summary
scope.capabilities.pixel_size_um           # raw per-installation um/pixel, or None if the scope cannot report it (unknown camera / no declared optics). For an objective-adjusted effective um/pixel, call common_utils.get_pixel_size(focal_length, binning_size).
scope.capabilities.lens_focal_length_mm    # tube-lens focal length, mm, or None if the scope cannot report it
```

```python
# Camera diagnostic snapshot. Returns dict with model, resolution,
# pixel_format, gain, exposure_ms, max_gain, max_exposure_ms,
# temperatures (Celsius), and per-field error strings when a probe
# fails. Returns {'connected': False} when no camera is active.
info = scope.diagnostics.get_camera_diagnostic_info()

# Camera temperature sensors. Returns dict {sensor_name: degC}; empty only
# for a camera with no temperature sensor (capabilities.camera_reports_temperature
# False); None when no camera is active. A read that fails on an active camera
# raises HardwareError: neither {} nor None ever stands for a failure.
temps = scope.diagnostics.get_camera_temperatures_degc()

# The camera's link, live: {'transport': 'USB3' | 'GigE' | 'USB2',
# 'link_speed', 'link_speed_unit', 'packet_size_bytes', 'inter_packet_delay'},
# each None where the camera does not report it (packet size and delay are
# GigE's). The speed is in the unit the camera declares, never converted:
# Basler's varies by model; the FX2's is 'Mbps'.
# None when no camera is active; a failed read raises HardwareError.
link = scope.diagnostics.get_camera_link_info()

# Throughput and latency characterization. Both run through the
# PRODUCTION capture path, so what they measure is what a real run gets;
# both take an optional progress_cb and return a dict of results. These
# occupy the camera for their duration -- do not start one while a run
# or recording is live.
results = scope.diagnostics.run_camera_bandwidth_test(num_frames=200, timeout_s=60.0)
results = scope.diagnostics.run_grab_lifecycle_benchmark(num_cycles=100, vary_settings=False)

# Cross-host / cross-camera / cross-firmware diagnostic probe.
# Captures camera identity, current config, temperatures, and stream
# stats deltas over duration_s, stamped with the active camera SDK
# (Basler pylon, IDS peak, ...). Writes JSON to data/camera_probe/.
# A driver that does not implement the probe returns the driver's
# {'supported': False, ...} shape unchanged. Does NOT change grab state.
probe = scope.diagnostics.run_pylon_diagnostic_probe(
    duration_s=3.0, drain_camera_side_errors=True,
)

# Engineering-mode firmware diagnostic commands. Routes through the
# canonical driver path (Rule 13 logging, Rule 14 error visibility).
# target is 'led' or 'motor'.
# When no reply came back the answer is a stand-in string ('Board not
# connected', 'None' for a timed-out read, 'No response', or 'Error: ...');
# is_board_reply() tells them from a reply. Check get_led_info()['command_set']
# before sending an LED command the board may not carry.
from modules.lumascope_api.diagnostics import is_board_reply
resp = scope.diagnostics.send_diagnostic_command('led', 'INFO')
lines = scope.diagnostics.send_diagnostic_command_multiline(
    'led', 'SELFTEST', timeout_s=60,
)

# Motor-board driver / fan diagnostics (already on
# DiagnosticsAPI pre-Phase-5; documented here for completeness).
status = scope.diagnostics.read_motor_drv_status('Z')       # int register or None
rpm = scope.diagnostics.read_motor_fan_rpm()              # RPM or None
ok = scope.diagnostics.set_motor_fan_duty(50)              # bool

# LED engineering-mode handshake (FACTORY / Y / Q with post-Q drain).
# Use these in place of open-coded send_diagnostic_command sequences.
ok = scope.diagnostics.enter_led_engineering_mode(timeout_s=5.0)
currents = scope.diagnostics.read_led_currents_ma()   # {channel: mA or None}; v2, in engineering mode
scope.diagnostics.exit_led_engineering_mode()
```

---

## scope.capabilities

`scope.capabilities` is a `ScopeCapabilities` dataclass populated at connect time. **Use this to learn what the connected hardware can do** — don't hardcode axis lists, LED channel counts, or camera caps.

```python
caps = scope.capabilities

# Motion
caps.axes                       # ('X', 'Y', 'Z', 'T') on LS850T; ()         on LS620
caps.has_focus                  # True if Z is motorized
caps.has_xy_stage               # True if X/Y are motorized
caps.has_turret                 # True if the turret axis is present
caps.model                      # e.g. 'LS850T'; the layer identity's model, '' if none

# LED
caps.led_channels               # e.g. (0, 1, 2, 3) for FX2 scopes; (0..5) for RP2040; None when the scope came up without its LED board
caps.led_colors                 # e.g. ('BF', 'Blue', 'Green', 'Red') — the layers THIS model lights, from its identity, board or not
caps.led_max_ma                 # per-channel current cap; None when the scope came up without its LED board
caps.has_firmware_stim          # firmware-timed stim support on the LED board

# Optics -- resolved from the first real source, never a hardcoded default:
#   motorconfig.json Optics (LS820/850/850T) -> scopes.json Optics (Classic)
#   -> camera SDK-reported pitch (pixel size only) -> None
caps.pixel_size_um              # um/pixel, or None if the scope cannot report it
caps.lens_focal_length_mm       # tube lens focal length mm, or None if unavailable

# Camera
caps.camera_model               # 'MT9P031-LS620', 'daA3840-45um', etc.; None if no camera or it did not say
caps.camera_serial_number       # the camera's serial number, or None
caps.camera_timestamp_tick_hz   # the camera's frame-timestamp clock in Hz, or None (no timestamps)
caps.is_color_native            # True for color-native sensors; False for mono-native (default)
caps.native_bit_depth           # 8 (e.g. IDS) or 16 (uint16 container; holds 12/16-bit native)
caps.camera_supports_auto_gain
caps.camera_supports_auto_exposure
caps.camera_supports_conversion_gain_mode
caps.camera_supports_line_noise_reduction
caps.camera_supports_black_level  # set_black_level offered; get_black_level reads on any camera that reports one
caps.camera_pixel_formats       # e.g. ('Mono8',) or ('Mono8', 'Mono12')
caps.camera_binning_sizes       # e.g. (1, 2, 4)
caps.camera_max_frame_size      # (width, height) unbinned, in pixels: the smallest of the sensor the profile
                                # documents, the camera's own maximum and the model's (data/scopes.json
                                # MaxFrame); None if none of them is known
caps.camera_analog_gain_max_db  # dB: the most gain applied before a digital stage (above it the camera
                                # multiplies digitised values). The profile's analog maximum, or the camera's
                                # live maximum where it has no digital stage; None if neither is known.
                                # The whole range, analog and digital, is scope.imaging.max_gain_db_cached
caps.camera_reports_temperature # the camera has a temperature sensor (probed at connect); False:
                                # get_camera_temperatures_degc answers {}
# Exposure ceiling: scope.imaging.max_exposure_ms_cached (ms; None if no camera) -- see scope.imaging
```

Important consequences:

- **`camera_max_frame_size` is `None` when no camera is connected** (or none of its sources answered). Check it before using it as a `scope.imaging.set_frame_size(w, h)` target, and divide by the binning in force; `set_frame_size` returns `None` (no-op) when no camera is active. With a live camera it returns the DELIVERED geometry and raises `CameraSettingRejected` if the apply is refused.
- **LED channel count varies by scope, and not only by driver family.** An LS620 (FX2 driver) exposes 4 channels (`BF`, `Blue`, `Green`, `Red`); an **LS560, same driver family, exposes 2** (`BF`, `Green`); RP2040-based scopes expose 6 (`BF`, `PC`, `DF`, `Blue`, `Green`, `Red`). Don't iterate over a hardcoded list — iterate over `caps.led_colors`.
- **Some scopes have no motor at all.** LS560/LS620 have `caps.axes == ()`. Calling `scope.motion.move_absolute('X', …)` against such a scope raises `HardwareCommandRefusedError` (`axis_absent`, `MissingPart.MOTORS`), as a Z-only scope does for X (`MissingPart.X`); hide motion controls based on `caps.has_xy_stage`, `caps.has_focus` and `caps.has_turret`.
- **Travel limits come from `scope.motion.get_axis_limits(axis)`**, read-only, for present axes; check `caps.has_xy_stage` (or `axis in caps.axes`) before asking about X/Y. A run whose steps lie outside these limits is refused before it starts (reason `positions_outside_travel`), naming each step and the axis; X/Y are judged on the protocol's plate at `scope.runtime_state.get_stage_offset()`, the offset the run moves with. A run whose steps need an axis the scope does not have is refused as `positions_unreachable`.

---

## scope.io

**Reserved.** Not populated in LumaViewPro 4.0.x.

The `scope.io` sub-API is named in the locked sub-API decomposition per `docs/PLUGIN_API_DESIGN_2026-05-09.md` §6.6. It will document future I/O surfaces (trigger devices, USB-to-IO trigger boards, external sync) once those surfaces ship; the feature flags that gate them will ride `scope.runtime_state` when they exist.

---

## scope.runtime_state

Mutable counterpart to `scope.capabilities`: the runtime-mutable user configuration (labware, objective, turret, stage).

Firmware versions are a live query: `scope.diagnostics.get_motor_info()['firmware_version']` / `scope.diagnostics.get_led_info()['firmware_version']`.

---

## modules

The `modules/` package holds helpers that ride alongside the API surface but are not sub-API methods. Two patterns:

- **Take `scope` as first argument**: orchestration helpers that compose sub-API calls (image-save, composite capture, protocol runner).
- **Pure functions**: stateless utilities that take frame arrays or geometry parameters (coord transformations, optical calculations, focus scoring).

### Image saving (`modules.image_save`)

Image-save helpers are free functions in `modules.image_save` (extracted
from the Lumascope class in Wave 7 Phase 6, 2026-05). Each function
takes the `scope` (a `Lumascope` instance) as its first argument; the
remaining arguments are the per-call settings:

```python
from modules.image_save import save_image

save_image(
    scope,
    array=image,
    save_folder='/path/to/output',
    file_root='experiment1',
    append='_BF_A1',
    channel='BF',                          # what was imaged -- recorded as identity
    false_color_on=False,                  # how it is displayed -- never recorded as identity
    tail_id_mode='increment',              # auto-number files
    output_format='TIFF',                  # 'TIFF' or 'OME-TIFF'
    save_encoding='right_aligned',         # from the image-mode config layer
    significant_bits=scope.imaging.capture_frame_depth(image),
    objective_id=objective_id,             # the objective the frame was taken with (read at capture)
    plate_x_mm=60.0, plate_y_mm=40.0,      # plate position the file records (mm)
    stage_z_um=5000,                       # stage Z (µm)
)
```

The full set of free functions in `modules.image_save`:

| Function | Purpose |
|---|---|
| `save_image(scope, array, ...)` | Save a numpy array to TIFF / OME-TIFF with metadata. |
| `prepare_image_for_saving(scope, array, ...)` | Flip / bit-convert / build metadata + path; returns `{'image', 'metadata'}`. |
| `generate_image_metadata(scope, channel, plate_x_mm, plate_y_mm, stage_z_um, *, objective_id)` | Build the TIFF metadata dict for the capture settings + position; the scale comes from `objective_id`, the objective the frame was taken with. |
| `generate_image_save_path(scope, save_folder, ...)` | Generate the next unused file path under `tail_id_mode`. |
| `get_next_save_path(scope, path)` | Increment the trailing numeric ID on an existing path. |

### Image utilities (`modules.image_utils`)

Boundary helpers that ride alongside the mono-native pipeline. Two
patterns matter to L2 callers:

```python
from modules import image_utils

# Map a mono frame to RGB false-color at the display / encode boundary.
# Use this when you have a mono fluorescence frame from get_image() and
# need a 3-channel array for display, video encode, or a downstream
# tool that expects RGB. Mono pipeline saves do NOT call this -- the
# layer is recorded as TIFF metadata instead.
rgb = image_utils.mono_to_rgb_falsecolor(mono_frame, layer='Blue')
# layer in {'Blue', 'Green', 'Red', 'BF', 'Lumi', ...}
# Returns 3-channel ndarray, same dtype as input.

# Read a TIFF and collapse legacy 3-channel false-color-replica files
# to mono on the fly. Use this when reading any TIFF that may have
# been written by a pre-mono-native LumaViewPro: the 3-channel files
# with one populated channel auto-collapse to 2D mono; true color
# composites (multiple non-zero channels) pass through unchanged.
img = image_utils.read_tiff_with_legacy_collapse(path)
# Returns 2D mono ndarray for mono and collapsed-legacy files;
# 3D RGB ndarray for real color composites.
```

The save pipeline emits mono fluorescence TIFFs with layer metadata
in the TIFF ImageDescription field; the legacy reader bridges that
to consumers that previously assumed a 3-channel shape. FIJI, MATLAB
``imread``, and tifffile all handle mono 2D natively; the false-
color is purely a display-time concern.

Full-pixel-depth frames store raw, right-aligned sensor values (a 12-bit
frame is ``0..4095``) and declare the true depth in the OME-TIFF
``SignificantBits`` tag (e.g. ``SignificantBits=12`` inside a 16-bit
container). To render or scale such a file to 8-bit, divide by
``(1 << SignificantBits) - 1`` -- treating the values as full 16-bit will
render a 12-bit frame ~16x too dark. A sum's tag is the bits it can reach
(``SignificantBits=14`` for four 12-bit frames, 10 for four 8-bit ones). ``image_utils.read_tiff_significant_bits``
returns the tag (falling back to the container width for older files that were
left-justified into the 16-bit range and carry no payload-depth tag).

### Coordinate transformations (`modules.coord_transformations`)

```python
from modules.coord_transformations import CoordinateTransformer
ct = CoordinateTransformer()

# Stage µm → plate mm (top-left origin)
plate_x, plate_y = ct.stage_to_plate(
    labware=labware_obj, stage_offset=offset, sx=60000, sy=40000,
)

# Plate mm → stage µm
stage_x, stage_y = ct.plate_to_stage(
    labware=labware_obj, stage_offset=offset, px=50.0, py=30.0,
)
```

`labware` is a `LabWare` object loaded from `data/labware.json` via `WellPlateLoader`, not a raw dict. `stage_offset` is a dict like `{'x': 0.0, 'y': 0.0}`.

### Optical calculations (`modules.common_utils`)

```python
import modules.common_utils as common_utils

# Pixel size (µm per pixel)
px_um = common_utils.get_pixel_size(focal_length=4.78, binning_size=1)

# Field of view (µm)
fov = common_utils.get_field_of_view(
    focal_length=4.78,
    frame_size={'width': 2048, 'height': 2048},
    binning_size=1,
)
# Returns: {'width': ..., 'height': ...} in µm, or None if scale is unknown
```

These helpers read `scope.capabilities.pixel_size_um` / `scope.capabilities.lens_focal_length_mm` from the active scope. Both return `None` when there is no active scope, or when the scope cannot report its optics (unknown camera, no declared optics) — `get_pixel_size` and `get_field_of_view` then return `None` rather than an invented scale, and callers degrade honestly (no scale bar, no field of view, no `PhysicalSizeX`). There is deliberately no hardcoded fallback: a guessed pixel size is written into every image and cannot be told from a measured one. Note: `capabilities.pixel_size_um` is the raw per-installation pixel pitch; for an effective µm/px adjusted for current objective + binning, call `common_utils.get_pixel_size(focal_length, binning_size)`.

### Composite capture (`modules.composite_builder`)

`build_composite()` composes multi-channel frames into a single false-color image. See the [Multi-channel composite](#multi-channel-composite) pattern under Common patterns for a runnable example.

```python
from modules.composite_builder import build_composite
```

### Autofocus (`modules.autofocus_functions`)

`focus_function(image=...)` computes a Brenner-gradient focus score from a frame array. Pure function; no scope state needed.

```python
from modules.autofocus_functions import focus_function

score = focus_function(image=frame, skip_score_logging=True)
```

Used by autofocus iteration code paths (was previously available as `scope.compute_focus_score(image)`; retired in Wave 7 Phase 7 per the rule that frame-analysis functions are pure helpers, not API methods).

### Protocol (`modules.protocol`)

`Protocol.from_file(...)` loads multi-step acquisition sequences. See the [Headless protocol run](#headless-protocol-run) pattern for a runnable example.

```python
from modules.protocol import Protocol
```

---

## plugin platform reference

Plugin platform spec and live-processing tutorial both live alongside LumaViewPro.

- **Design**: `docs/PLUGIN_API_DESIGN_2026-05-09.md` — the locked platform spec (PluginSpec, namespaces, registry contracts, loading sequence).
- **Plugin tutorial**: `docs/PluginTutorial.md` — the plugin shape and lifecycle, worked for `ctx.plugins.post_processing`.
- **Namespaces (4.x)**: `ctx.plugins.ui`, `ctx.plugins.post_processing`, `ctx.plugins.live_processing`, `ctx.plugins.rest`.

A worked plugin example ships in `etaluma-engineering/`; see its `pyproject.toml` `entry_points` for how a plugin declares itself.

---

## REST surface reference

REST is **not implemented**. There is no server, no endpoints, and no wire
format — nothing in this repo answers HTTP. Endpoints will be documented here
when REST actually ships, against the real implementation.

Nothing about a future REST surface is specified anywhere in this document. An
earlier revision carried a sketch of endpoint shapes and a MATLAB client; both
predated the 4.0 API-surface work, were never built, and were removed once they
started being read as a contract that constrained that work. `git log` has them
if the ideas are ever wanted.
---

## Common patterns

### Basic capture

```python
from modules.scope_session import ScopeSession

# create() builds the scope and brings it up from the user's settings: the
# objective, plate, offset and scale bar a capture uses are those settings.
session = ScopeSession.create(ScopeSession.load_user_settings('.'))
scope = session.scope
scope.motion.home()                        # returns once the home has established the reference
scope.imaging.set_exposure_ms(50)
scope.imaging.set_gain_db(5.0)

scope.motion.move_absolute('X', 60000)
scope.motion.move_absolute('Y', 40000)
scope.motion.move_absolute('Z', 5000)

from modules.image_save import save_image

scope.illumination.led_on('BF', 100)
objective_id, _ = scope.runtime_state.resolve_current_objective()  # the objective this frame is taken with
image = scope.imaging.capture_and_wait()
scope.illumination.leds_off()

save_image(
    scope,
    array=image, save_folder='./output',
    file_root='capture', append='_BF',
    channel='BF', false_color_on=False,
    save_encoding='right_aligned',
    significant_bits=scope.imaging.capture_frame_depth(image),
    objective_id=objective_id,
    output_format='TIFF', x=60000, y=40000, z=5000,
)
session.shutdown()
```

### Multi-channel composite

A composite is a run kind, not a loop you write. One step per channel set
to acquire an image, each at its own stored focus, followed by the merge --
through the same engine as every other run, so it produces a run directory
and an execution record like any scan.

```python
runner = session.create_protocol_runner()

# Which channels take part, their illumination, exposure, gain and blend
# thresholds all come from the settings snapshot. Set them there rather
# than passing them here, so a headless caller and a GUI click run the
# same composite.
outcome = runner.run_composite(sequence_name='composite')
print(f'merged composite: {outcome.artifact_path}')
if outcome.status == 'incomplete':
    print(f'merged without: {[failed.step_name for failed in outcome.captures.failed]}')
```

`run_composite` blocks until the merge settles and returns the run's
outcome, the merged file's path on it, so a missing artifact cannot be
mistaken for a success. A composite merged from fewer channels than it
was asked for reports `status='incomplete'` and names the failed channels
in `captures.failed`. To
launch one without waiting, call `runner.start_composite(...)`, which
returns the run's merge outcome to wait on or ignore. With no
`parent_dir` the run lands under `Manual/Composites` in the live folder,
where the Composite button puts it.

It raises `ProtocolRunRefusedError` when the run is refused before
anything is committed -- fewer than two channels set to acquire an image,
a rival run holding the scope, or files still draining -- and
`CaptureError` when the run happened but produced no composite. The
refusal's `str()` is the sentence written for a person, as for any run.

**`CaptureError.reason` is a failure code, not a refusal.** It names what
went wrong after the run committed (`write_batch_timeout` -- the run's
images did not finish writing within the merge's bound --
`write_batch_abandoned` -- some were given up on by a writer recovery or a
shutdown -- `write_batch_not_taken` -- some never reached the file writer,
which was stuck or had stopped taking work -- `write_batch_save_failed` --
saving some failed on disk -- `write_batch_disk_full` -- some were refused
because the save drive was nearly full -- `write_batch_video_unfinished` --
a video step's file did not finish --, `merge_failed`,
`aborted`, the composite builder's own codes such as `no_data`,
`excluded_inputs` or `post_processing_incomplete`, ...) and is not a member of the refusal family: a refusal means
nothing changed, while a `CaptureError` means the run ran and did not
produce the artifact.

To merge frames you already hold in memory, `build_composite` is the
underlying helper; it accepts fluorescence keys `'Red'`, `'Green'`,
`'Blue'`, `'Lumi'`, takes `significant_bits` for the input depth, and reads
`brightness_thresholds` on the OUTPUT 8-bit scale (absolute values 0-255,
not percentages).

### Z-stack

```python
from modules.image_save import save_image

z_start, z_end, z_step = 4000, 6000, 50    # µm

scope.illumination.led_on('BF', 100)
z = z_start
while z <= z_end:
    scope.motion.move_absolute('Z', z)
    objective_id, _ = scope.runtime_state.resolve_current_objective()  # the objective this frame is taken with
    image = scope.imaging.capture_and_wait()
    save_image(
        scope,
        array=image, save_folder='./zstack',
        file_root='z', append=f'_{int(z)}',
        channel='BF', false_color_on=False,
        save_encoding='right_aligned',
        significant_bits=scope.imaging.capture_frame_depth(image),
        objective_id=objective_id,
        output_format='TIFF', z=z,
    )
    z += z_step
scope.illumination.leds_off()
```

### Well-plate scan

```python
from modules.coord_transformations import CoordinateTransformer
from modules.image_save import save_image
ct = CoordinateTransformer()

wells = [('A1', 10.0, 20.0), ('A2', 19.0, 20.0), ('A3', 28.0, 20.0)]

scope.illumination.led_on('BF', 100)
for well_name, px, py in wells:
    sx, sy = ct.plate_to_stage(labware=labware_obj, stage_offset=offset, px=px, py=py)
    scope.motion.move_absolute('X', sx)
    scope.motion.move_absolute('Y', sy)

    objective_id, _ = scope.runtime_state.resolve_current_objective()  # the objective this frame is taken with
    image = scope.imaging.capture_and_wait()
    save_image(
        scope,
        array=image, save_folder='./scan',
        file_root=f'{well_name}_BF',
        channel='BF', false_color_on=False,
        save_encoding='right_aligned',
        significant_bits=scope.imaging.capture_frame_depth(image),
        objective_id=objective_id,
        output_format='TIFF', x=sx, y=sy,
    )
scope.illumination.leds_off()
```

### Headless protocol run

```python
from modules.scope_session import ScopeSession
from modules.protocol import Protocol

session = ScopeSession.create(ScopeSession.load_user_settings('.'), simulate=True)    # simulated, configured, executors running; '.' must be an LVP root. simulate=False for hardware

protocol = Protocol.from_file(
    file_path='./my_protocol.tsv',
    tiling_configs_file_loc='./data/tiling.json',
)

runner = session.create_protocol_runner()
pending = runner.run_single_scan(
    protocol,
    image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
)
result = pending.wait(timeout_s=300)
print(result.status, result.reason, result.message)

session.shutdown()          # the factory built this scope, so shutdown() disconnects it
```

---

## Simulated mode

Use for development, CI, and unit tests without hardware.

```python
scope = Lumascope(simulate=True)
scope.imaging.start_streaming()   # a scope you build yourself streams once started

# All API calls work identically:
scope.illumination.led_on('Blue', 200)
scope.motion.move_absolute('Z', 5000)
image = scope.imaging.get_image()
```

A camera connects configured but not grabbing, simulated or real, so a `Lumascope` you build yourself returns no frame from `get_image()` until `scope.imaging.start_streaming()`. A scope brought up through `ScopeSession.create()` is already streaming.

**Only in `simulate=True`**: `set_timing_mode('fast')` lets simulator tests run faster by skipping artificial serial / motor / camera delays. Same private-driver access pattern: timing-mode control is a simulator test-infrastructure feature, not an L2 surface.

```python
scope._led_driver.set_timing_mode('fast')
scope._motion_driver.set_timing_mode('fast')
scope._camera_driver.set_timing_mode('fast')
```

These attributes only exist on the simulated drivers. Don't call them on a real-hardware `Lumascope` — you'll get `AttributeError`.

---

## Protocol file format

Tab-separated file defining multi-step acquisition sequences.

```
LumaViewPro Protocol
Version	5
Period	1.0
Duration	0.002778
Labware	96 well microplate
Capture Root

Steps
Name	X	Y	Z	Auto_Focus	Color	...
A1_BF	60000	40000	5000	False	BF	...
A1_Green	60000	40000	5000	False	Green	...
```

**Step fields:**

| Field | Type | Description |
|---|---|---|
| Name | string | Step label (e.g. `A1_BF`) |
| X, Y, Z | float | Position in µm |
| Auto_Focus | bool | Run autofocus at this step |
| Color | string | `Blue`, `Green`, `Red`, `BF`, `PC`, `DF`, `Lumi` |
| False_Color | bool | Apply false-color mapping |
| Illumination | float | LED current in mA |
| Gain | float | Camera gain in dB |
| Auto_Gain | bool | Enable auto-gain |
| Exposure | float | Exposure time in ms |
| Sum | int | Frame averaging count (≥1) |
| Objective | string | Must match `data/objectives.json` |
| Well | string | Well label (e.g. `A1`) |
| Acquire | string | `image` or `video` |

Consult `Protocol.from_file` in `modules/protocol.py` for the canonical field list — additions happen over time.

---

## Color channel reference

```python
from modules.common_utils import ColorChannel

ColorChannel.Blue   # 0  — blue-EMISSION fluorescence   (stock excitation 405 nm)
ColorChannel.Green  # 1  — green-EMISSION fluorescence  (stock excitation 488 nm)
ColorChannel.Red    # 2  — red-EMISSION fluorescence    (stock excitation 589 nm)
ColorChannel.BF     # 3  — brightfield (white LED)
ColorChannel.PC     # 4  — phase contrast (on scopes with separate PC hardware)
ColorChannel.DF     # 5  — darkfield
ColorChannel.Lumi   # 6  — luminescence (all LEDs off, sensitive mode)
```

**A channel name is the colour you SEE, not the colour that excites it.** `Blue` is a blue-emitting dye excited by 405 nm violet light, `Green` a green-emitting dye excited by 488 nm, `Red` a red-emitting dye excited by 589 nm. The layer's `excitation_nm` field carries the excitation wavelength, and the UI marks it `Ex` for the same reason: a bare "Green 488 nm" invites reading 488 as the emission, which writes silently wrong metadata into your own data.

**Fluorescence excitation wavelengths depend on the installed filterset** — the stock filterset is 405 / 488 / 589 nm, but OEM customers may have custom filtersets at different wavelengths.

**Not every scope has every channel, and two models in the same family differ.** Always check `scope.capabilities.led_colors` before using a color — an LS620 exposes `{'BF', 'Blue', 'Green', 'Red'}`, while an **LS560 exposes only `{'BF', 'Green'}`**. Phase contrast on those models is brightfield with a mechanical phase slider installed, not a separate illumination channel.

---

## Appendix A: Internal serial-protocol interfaces (firmware tooling only)

This appendix documents direct serial commands used by firmware update tools, board bring-up scripts, and factory calibration. **These are not intended for integration code** — they bypass safety limits, depend on chip-internal register semantics that can change across firmware versions, and can leave the hardware in unsafe states if misused. Application code should stay at the ScopeSession or Lumascope sub-API layer.

<details>
<summary>Show internal interfaces</summary>

### Direct board drivers

```python
from drivers.ledboard import LEDBoard
from drivers.motorboard import MotorBoard
from modules.path_utils import read_installation_file, resolve_data_file

led = LEDBoard()                           # auto-detect by VID:PID
led.exchange_command('LED3_200')           # set BF LED to 200 mA
led.exchange_command('LEDS_OFF')

# The motor driver takes the shipped defaults (travel limits, microsteps
# per mm) as a required argument; a missing or unusable file raises
# InstallationFileError, naming it.
defaults = read_installation_file(resolve_data_file('motorconfig_defaults.json'))
motor = MotorBoard(motorconfig_defaults=defaults)  # auto-detect by VID:PID
motor.exchange_command('HOME')
motor.exchange_command('TARGET_WZ682666')  # move Z (µsteps)
pos = motor.exchange_command('ACTUAL_RZ')
```

### Connection parameters

| Parameter | LED board | Motor board |
|---|---|---|
| VID:PID | 0x0424:0x704C | 0x2E8A:0x0005 |
| Transport | UART via USB hub bridge, 115200 baud | USB CDC native |
| Line ending (send) | `\r\n` | `\n` |
| Line ending (recv) | `\r\n` | `\n` |
| Command timeout | 100 ms default | 5 s default (homing: 15–30 s) |

### Raw REPL (firmware file transfer)

```python
motor.enter_raw_repl()
motor.repl_list_files()
content = motor.repl_read_file('motorconfig.json')
motor.repl_write_file('main.py', new_source)
motor.exit_raw_repl()
```

`SerialBoard` (the shared base class) implements raw REPL for both boards.

### LED board application commands (safe-mode)

| Command | Description |
|---|---|
| `INFO` | Board info (firmware version, calibration status, heap) |
| `LEDS_ENT` / `LEDS_ENF` | Enable / disable LED driver |
| `LEDS_OFF` | Turn off all LEDs |
| `LED{ch}_{mA}` | Set channel 0–7 to `mA` (float ok: `LED3_200`, `LED0_0.5`) |
| `LED{ch}_OFF` | Turn off channel |
| `LEDREAD{ch}` | Read I_SENS + LED_K ADC feedback |

Engineering-mode commands (`FACTORY`, `RAW…`, `ADCREAD`, `CALIBRATE`, `CALSAVE`, `CALCLEAR`, `SELFTEST`, `I2CSCAN`, `FWUPDATE`) bypass safety limits and are **not documented here** — they exist for factory bring-up and firmware development only.

### Motor board application commands

| Command | Description |
|---|---|
| `INFO` / `FULLINFO` | Firmware and board info |
| `HOME` / `ZHOME` / `THOME` | Home all / Z only / turret only |
| `CENTER` | Move stage to center |
| `STOP` | Stop all motors immediately |
| `TARGET_W{axis}{steps}` | Set target position (µsteps) |
| `TARGET_R{axis}` | Read target position |
| `ACTUAL_R{axis}` | Read current position |
| `STATUS_R{axis}` | Read status register (32-bit) |
| `VOLTAGE` | Rail status |
| `CURRENT` | Per-axis motor current telemetry |

Axes: `X`, `Y`, `Z`, `T`. Position conversion (µsteps ↔ µm) is in `motorconfig.json`; the API takes and answers micrometres, and an axis's travel is `scope.motion.get_axis_limits(axis)`, so a caller never reads that file.

During homing, `STOP` aborts. `INFO`, `ACTUAL_R`, `STATUS_R`, `VOLTAGE` respond normally. Other commands return `BUSY`.

Direct SPI access to the TMC5072 (register-level motor configuration) and the associated status-register bit semantics are intentionally omitted — those are firmware-internal.

</details>


---

## Changelog

Post-freeze record of every change to the L2-callable surface (methods documented above). Entries land in the SAME commit as the underlying change. Pre-`4.0.0` (the freeze trigger) is fluid by design and not recorded here; consult `LVP_4.0.0_CHANGELOG.md` for pre-freeze prose history.

Entry format:

| Version | Date | Type | Method / surface | Change |
|---|---|---|---|---|

- **Type**: `additive` (new optional param / new return-dict key / new method) — no version bump beyond patch.
- **Type**: `behavior-change` — requires minor version bump.
- **Type**: `rename` / `removal` — requires deprecation cycle (old name retained with `FutureWarning`, retired in next major) OR a major version bump.

(No entries yet -- 4.0.0 has not shipped. Pre-freeze structural changes do not appear here.)
