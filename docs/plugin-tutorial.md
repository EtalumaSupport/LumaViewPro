# LumaViewPro -- Plugin Tutorial

## PRE-RELEASE API

The plugin platform documented in this file is the 4.x plugin
contract: what this tutorial teaches freezes with the API when `4.0.0`
ships, and may change without a deprecation cycle until then.
`ctx.plugins.post_processing` operates on saved files. `ctx.plugins.ui`
has one mount point, `left_sidebar.accordion`.
`ctx.plugins.live_processing` is a per-frame listener on the live camera
stream. `ctx.plugins.rest` is name-reserved and raises on registration;
`ctx.plugins.rest.register` is outside the freeze -- read **Lifecycle
and namespaces** below before designing around it.

If you are shipping a plugin against this contract before `4.0.0`,
contact Etaluma support so we can consult you before structural
changes.

---

## What a plugin is

A LumaViewPro plugin is a Python package that the host discovers at
startup via the `lvp.plugins` entry-points group. Each plugin exposes
a module-level `PluginSpec`, a `register(ctx)` function, and an
optional `unregister(ctx)`. `register(ctx)` calls one or more
`ctx.plugins.<namespace>.register(...)` methods to attach the
plugin's handlers; the host wraps the call in try/except so a bad
plugin cannot crash the app. `ctx` is the `ScopeSession` hosting the
plugin: the same session the GUI, REST and scripts drive, so a plugin
reaches the instrument through `ctx.scope` and the session's members
(`docs/lumascope-skills.md`).

Four namespaces are defined for 4.x:

| Namespace | Purpose | State in this release |
|---|---|---|
| `ui` | Adds widgets at named mount points | live; one mount point (`left_sidebar.accordion`) |
| `post_processing` | Operates on saved capture folders | live; this tutorial's subject |
| `live_processing` | Per-frame listeners on the live camera stream | live; see **Live frame listeners** below |
| `rest` | HTTP sub-routers mounted under `/plugins/<name>/` | reserved; raises on registration |

This tutorial covers `post_processing` end-to-end. The other three
are under **Lifecycle and namespaces**: **UI mounts**, **Live frame
listeners** and **The reserved namespace**.

---

## Hello-world post-processor

The smallest possible plugin: a processor that walks an input
directory, tags every TIFF it finds, and writes a one-line summary
into the output directory. No image edits; the goal is to see the
contract end-to-end.

### Directory layout

```text
hello_postproc/
    pyproject.toml
    hello_postproc/
        __init__.py
```

### `hello_postproc/__init__.py`

```python
"""Hello-world post-processor for LumaViewPro.

Registers a processor that lists the TIFFs in input_dir and writes a
summary.txt into output_dir. Demonstrates the plugin contract; not a
useful processor on its own.
"""
from __future__ import annotations

import os
from typing import Any

from lumascope import PluginSpec, ProcessorResult


__version__ = "0.1.0"


spec = PluginSpec(
    name="hello_postproc",
    version=__version__,
    requires_lvp_version=">=4.0.0",
    description="Lists TIFFs in a capture folder and writes a summary.",
    capabilities=("scope.imaging",),
    author="Your Name",
    url="https://example.com/hello_postproc",
)


def _process(input_dir: str, manifest: dict, output_dir: str) -> ProcessorResult:
    """Processor callable. Signature is fixed by the post_processing namespace.

    Args:
        input_dir: Absolute path to the capture folder being processed.
        manifest: Capture manifest dict. Contents are protocol-specific;
            treat unknown keys as opaque.
        output_dir: Absolute path the plugin writes outputs into. The
            host creates this directory before invoking the processor.

    Returns:
        ProcessorResult naming the produced files plus a one-line
        user-facing message.
    """
    tiffs = []
    for entry in sorted(os.listdir(input_dir)):
        if entry.lower().endswith((".tif", ".tiff")):
            tiffs.append(entry)

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(f"hello_postproc v{__version__}\n")
        fh.write(f"input_dir: {input_dir}\n")
        fh.write(f"tiff_count: {len(tiffs)}\n")
        for name in tiffs:
            fh.write(f"  - {name}\n")

    return ProcessorResult(
        success=True,
        outputs=(summary_path,),
        message=f"Tagged {len(tiffs)} TIFF(s).",
    )


def register(ctx: Any) -> None:
    """Called once at app startup. Attach the processor to the registry."""
    ctx.plugins.post_processing.register(spec, _process)


def unregister(ctx: Any) -> None:
    """Optional. Called at app shutdown in reverse load order.

    Nothing to tear down for this plugin. Implementations that opened
    files, started threads, or registered listeners must release them
    here so a clean re-load works.
    """
    return None
```

### What each `PluginSpec` field means

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | Unique within a namespace. Duplicate -> `PluginRegistrationError`. |
| `version` | yes | Your plugin's version. Free-form; not parsed by the host. |
| `requires_lvp_version` | yes | PEP-440-ish requirement (`>=4.0.0`, `==4.0.0`, `~=4.0.0`). Host strips pre-release suffixes (`-beta8`) before comparing. |
| `description` | yes | One sentence saying what the plugin does. |
| `capabilities` | no | Tuple of dotted paths the plugin uses (`scope.imaging`, `scope.camera`). A declaration only: not read, and not sandbox-enforced, in 4.x. |
| `author` | no | Free-form. |
| `url` | no | Free-form. |
| `auto_run_on_protocol_complete` | no | `post_processing` only. `True` hands your processor the folder of every Full Protocol run; see **After a protocol**. Defaults to `False`. |

`PluginSpec` is a frozen dataclass; attempting to mutate a field
after construction raises.

The tech-support report records each loaded plugin's `name`, `version`
and namespace, each plugin that did not load with its reason, and the
errors a loaded plugin raised while the host called it.

### What `ProcessorResult` means

| Field | Meaning |
|---|---|
| `success` | True if the processor completed. When the host ran it (**After a protocol**), False is reported to the person as your plugin's failure, with your `message`. |
| `outputs` | Tuple of absolute paths of the files the processor wrote. When the host ran it, its log says how many and the folder they share. |
| `message` | One-line summary for the person using the microscope; no internal IDs or exception class names. Logged when the host's run succeeds, shown when it fails. |

`ProcessorResult` is also a frozen dataclass.

---

## Project layout and registration

The host discovers plugins via the `lvp.plugins` entry-points group.
A minimum `pyproject.toml` for the `hello_postproc` package above:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "hello-postproc"
version = "0.1.0"
description = "Hello-world post-processor for LumaViewPro."
requires-python = ">=3.12"
authors = [{name = "Your Name", email = "you@example.com"}]

[project.entry-points."lvp.plugins"]
hello_postproc = "hello_postproc"

[tool.setuptools.packages.find]
where = ["."]
```

The single line under `[project.entry-points."lvp.plugins"]` is what
makes the plugin discoverable. The left side (`hello_postproc`) is
the entry-point name shown in error messages; the right side is the
importable module that exposes `spec`, `register`, and optionally
`unregister`.

Install into the same Python environment LumaViewPro runs in:

```text
pip install -e /path/to/hello_postproc
```

Restart LumaViewPro. At startup its session iterates entry points,
imports each one, reads its `spec`, checks `requires_lvp_version`
against the running LVP version, then calls `register(ctx)`. If any
step fails the plugin is skipped and the rest of the app keeps loading;
when discovery ends, every plugin that did not load is reported
together, once, each with its reason.

---

## Testing your plugin

LumaViewPro hands `register(ctx)` its session. A test hands it a stand-in
that holds only what a post-processor's `register` uses, a real
`PluginRegistry` from `lumascope`, so your registration runs against the
production registry code without LumaViewPro, Kivy or hardware. The
registry is built with no scope, which a post-processor needs none of. The
stand-in has nothing else, so a plugin that reaches for anything more fails
in your test with `AttributeError`.

Place this in `tests/test_hello_postproc.py`:

```python
"""Tests for hello_postproc. Run with: pytest tests/test_hello_postproc.py"""
from __future__ import annotations

import types

import pytest

from lumascope import PluginRegistry

import hello_postproc


@pytest.fixture
def ctx():
    """A fresh stand-in for the session per test: registrations do not leak across tests."""
    return types.SimpleNamespace(plugins=PluginRegistry(None))


def test_registers_in_post_processing_namespace(ctx):
    hello_postproc.register(ctx)
    assert "hello_postproc" in ctx.plugins.post_processing.names()


def test_processor_writes_summary(ctx, tmp_path):
    hello_postproc.register(ctx)
    processor = ctx.plugins.post_processing.get("hello_postproc")

    input_dir = tmp_path / "capture"
    input_dir.mkdir()
    (input_dir / "frame_001.tiff").write_bytes(b"")
    (input_dir / "frame_002.tif").write_bytes(b"")
    (input_dir / "notes.txt").write_bytes(b"")

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    result = processor(str(input_dir), {}, str(output_dir))

    assert result.success is True
    assert result.outputs == (str(output_dir / "summary.txt"),)
    summary = (output_dir / "summary.txt").read_text(encoding="utf-8")
    assert "tiff_count: 2" in summary


def test_unregister_does_not_raise(ctx):
    hello_postproc.register(ctx)
    hello_postproc.unregister(ctx)
```

Run them with the Python LumaViewPro's install script set up, which
imports `lumascope` from any folder.

---

## Lifecycle and namespaces

### Load

A host asks its session for plugins with `session.load_plugins()`;
LumaViewPro does at startup. A script's or a test's session that never
asks has none. For each entry point in `lvp.plugins`, and then for each
built-in:

1. Import the module. Import failure -> not loaded.
2. Read module-level `spec`. Missing or not a `PluginSpec` -> not loaded.
3. Check the spec: a `requires_lvp_version` the running LVP version
   does not meet -> not loaded.
4. Call `register(ctx)` with the session. Exception -> remove whatever
   it registered before the exception (a widget, a processor, a frame
   listener), then call `unregister(ctx)`, if the plugin has one, for
   its own cleanup -> not loaded.
5. Track the module so `session.unload_plugins()` can call its
   `unregister` at shutdown.

A plugin that did not load is recorded with its reason, for
`session.plugin_health()` and the tech-support report.

An installed plugin claiming a built-in's name keeps it; the built-in
is the one reported as not loaded.

Each step's failure leaves the rest of the app intact. The one report
when discovery ends says, for each plugin, `The "<name>" plugin did not
load: <reason>.`, then `The rest of LumaViewPro is unaffected.` -- if
you see that on startup, that's why.

### Unload

At app shutdown `session.unload_plugins()` walks the loaded list in
reverse and calls each plugin's `unregister(ctx)`; a session's
`shutdown()` does it first for a host that did not. Exceptions are
caught and logged at WARNING; shutdown is not blocked by a plugin's
teardown failure. It runs before the hardware is disconnected: a plugin
driving the scope stops that work here and waits, bounded, for it to
put the scope back.

Reverse order matters when plugin B was registered after plugin A
and depends on resources A exposes -- B comes down first.

### Settings change

No plugin is told of a settings change (plugin API level 9): the session
saves every accepted write itself, and `on_settings_changed` is not
called. A plugin that needs a setting reads it when it needs it
(`ctx.get_setting('video.max_fps')`). `PluginSpec.subscribes_to` is gone
(plugin API level 59): a spec that passes it raises `TypeError` where it
is built.

### After a protocol (optional)

If your post-processing spec sets `auto_run_on_protocol_complete=True`,
your processor is handed the folder of every Full Protocol run, whoever
started it, once its images are written and its hyperstack build has
ended. A run whose images are not all written is handed to no
processor. It runs on the post-processing lane, apart from the run: the
scope takes the next run while it works. From there it may call the
session's post-processing members; a blocking call to a camera, motion
or file member raises.

### UI mounts

`ctx.plugins.ui.register(spec, mount_point, builder)` adds a widget at
a named mount point. `mount_point` is a `UIMountPoint`, or its plain
string; the one mount point is `left_sidebar.accordion`
(`UIMountPoint.LEFT_SIDEBAR_ACCORDION`), and another name is refused
`ArgumentRefusedError` (`not_offered`) before anything registers.
`builder` takes no arguments and returns a Kivy `AccordionItem`. The
host calls it once the window's widget tree exists, not at
registration, and adds what it returns to the accordion; a builder
that raises is reported as your plugin's failure, and the rest of the
window still comes up.

```python
def register(ctx):
    ctx.plugins.ui.register(spec, 'left_sidebar.accordion', lambda: MyPanel(ctx))
```

The host does not lock a mounted widget while a run or a diagnostic
holds the scope. A plugin greys its own controls from
`ctx.add_run_state_listener` and `ctx.controls_lock.controls_locked`,
as the built-in regions do, so a control that must stay live during a
run, such as its own Stop, stays live.

### Live frame listeners

`ctx.plugins.live_processing.register(spec, handler)` adds `handler`
as a listener on every frame the camera delivers, live view and runs
alike. The registry forwards to `scope.imaging.add_frame_listener`
under your plugin's name, so the log names your plugin when the
handler runs over its budget or raises.
`ctx.plugins.live_processing.unregister(spec.name)` removes it.

```python
def register(ctx):
    def on_frame(frame):
        # Runs on the camera's thread: return quickly, and never write
        # to frame.image (draw on frame.image.copy()).
        ...

    ctx.plugins.live_processing.register(spec, on_frame)
```

The handler is called `handler(frame)` with the camera's `Frame`
record. The record's fields, the read-only frame, the per-frame time
budget and the auto-removal of a handler that overruns it or keeps
raising are in the guide's [Live frames](guide/python.md#live-frames). With no
camera connected, registration raises and nothing is registered.

### The reserved namespace

`ctx.plugins.rest` is reserved, and a plugin cannot add REST routes in
this release, so a shipped plugin registers none:
`ctx.plugins.rest.register(...)` raises `PluginRegistrationError`
(`rest_namespace_reserved`), and is outside the 4.0 freeze: it takes a
framework router, which has no published type, and what a plugin route
looks like is not settled. Don't catch and ignore the raise; let your
plugin fail to load, and serve what it needs through `post_processing`,
`live_processing` or `ui`. While a plugin is being developed,
`ctx.plugins.rest.enable_stub_mode()` records a registration and serves
nothing, so the rest of the plugin can be tried.

---

## The plugin API level

`session.plugin_api_level` says what
this LumaViewPro does that a plugin may rely on, within its major version.
A version cannot say it: many builds share one beta number. A change to
behaviour a plugin relies on raises the level by one. A plugin that needs a level checks
it at its own entry and refuses below it, naming both levels; a host older
than the level has no `plugin_api_level`, which reads as 0
(`getattr(session, 'plugin_api_level', 0)`). `PluginSpec.requires_lvp_version`
still states the major version.

Each level's row says what it added that a plugin may rely on; a level
whose change concerns only Etaluma's own tools says so.

| Level | Added |
|---|---|
| level 1 | `capabilities.camera_limits.analog_gain_db_max`; `capabilities.camera_reports_temperature`, and `get_camera_temperatures_degc` answering `{}` only with no sensor, `None` with no camera, and raising on a failed read. |
| level 2 | `imaging.get_black_level_range()`: the range `set_black_level` accepts for the current pixel format. |
| level 3 | With no LED board: `led_on` and a lighting transition raise `not_connected`, the LED state reads (`get_led_state`, `get_led_states`, `save_led_state`) answer `None`, and `capabilities.led_max_ma` / `led_channels` are `None`; `led_on` for an LED the model lacks raises `not_on_this_scope`. |
| level 4 | With no camera: every camera command raises `not_connected` (`MissingPart.CAMERA`) but an off whose end state holds; `get_gain_db`, `get_exposure_ms`, `get_width` and `get_height` answer `None`; after `disconnect()` the camera cache holds the no-camera values and `get_camera_temperatures_degc` / `get_camera_link_info` answer `None`. |
| level 5 | A plugin's `register`, `unregister` and `on_settings_changed` are handed the session (see "Lifecycle and namespaces"); `on_settings_changed` is told changes made since load; auto-run fires for every Full Protocol starter, on the post-processing lane, after the hyperstack build. |
| level 6 | `imaging.set_binning_size`, `set_pixel_format`, `set_conversion_gain_mode` and `set_line_noise_reduction`, and the Session's `set_image_mode`, `set_high_conversion_gain` and `set_line_noise_reduction`, return `None`: every failure raises, a camera gone before the body with `not_connected`; a toggle's mode on a camera without it raises `CameraSettingUnsupportedError`, its off is satisfied. |
| level 7 | Nothing a plugin outside Etaluma relies on. |
| level 8 | Every `Refusal` and `Quiet` type states its cause (`lumascope.RefusalCause`), as `cause` or, where its reasons differ, as `causes` keyed by reason: a plugin's own refusal type that states neither is refused at import, and one built with a reason its `causes` lacks raises `TypeError`. |
| level 9 | The session saves every accepted settings write itself (`ScopeSession.create(settings_file=...)`), and `save_settings`, `SettingsSaveRefusedError` and `on_settings_changed` are gone; `PluginSpec.subscribes_to` is accepted empty, and a spec naming keys is not loaded, its reason naming level 8 as the last that told a plugin of a settings change. |
| level 10 | LED state is by address: `get_led_state` (a layer name or an address) answers an `LedReading` (`address`, `name`, `state`, `illumination_ma`), `get_led_states` a tuple of them for every address of the board, `save_led_state` an `LedSnapshot`; the LED listener is called with `(address, name, state, illumination_ma)`; `get_led_state` of a layer the identity gives no LED raises `not_on_this_scope`; the drivers' `ch2color`, `color2ch`, `available_colors` and `led_ma` are gone, and the EL-0940 board addresses 0-7; an LED write the active lease does not permit raises `exclusive_activity_running`, and `led_on` always returns the commanded current. |
| level 11 | A run returns its `Operation` (`start_move_absolute`, `start_move_relative` and `start_home` return one too), and `ProtocolRunner`'s run members start it: `run_single_scan`, `run_protocol`, `run_autofocus`, `run_autofocus_all_steps` and `run_zstack` are `start_single_scan`, `start_protocol`, `start_autofocus`, `start_autofocus_all_steps` and `start_zstack`. |
| level 12 | A capture returns a frame or raises: `capture_and_wait` and `get_image` raise `CaptureFailedError` (`no_frame`, `deadline_expired`, `chunk_rejected` with its `setting`) for no frame and `HardwareCommandRefusedError` (`not_connected`, `camera_not_streaming`) for a camera gone or not streaming, and never answer None; `capture_failure_cause` and the failure flags of `last_capture_info` (`not_ready`, `drain_failed`, `deadline_expired`, `chunk_rejected`) are gone. |
| level 13 | A frame listener is called `cb(frame)` with the camera's read-only `Frame` record (`image`, `seq`, `timestamp`, `arrived_ns`, `significant_bits`, `chunks`), where it was called `cb(image, timestamp, chunks)`; the image a capture returns (`capture_and_wait`, `get_image`) is read-only. |
| level 14 | `imaging.judge_frame(frame)` answers whether a frame counts for each source's last recorded change (a `Judgment` per change, carrying its `Change` and the `Credit` of each frame counted toward it), and `frame_validity.count_frame` takes who counted (`by=Counter`); a capture's `frame_record` carries `judgments`, and its saved file `frame_validity`. |
| level 15 | `capabilities.camera_has_clock_latch`, and a capture's `frame_record` carries `clock_mapping`: the camera's clock mapped onto the host's steady clock beside the grab (`lumascope.ClockMapping`), None on a camera without a latch. |
| level 16 | Nothing a plugin outside Etaluma relies on. |
| level 17 | A capture's `frame_record` carries `saturated_fraction` (measured when `all_ones_check` is asked) and `dark` (judged when an LED is lit and `accept_dark` is not), each None when not measured, and answers `blown`; `last_capture_info` no longer carries `dark_saved`. A blown or dark frame is not logged as a warning by the capture; a run's outcome counts them (`captures.saturated`, `captures.dark`, each a tuple of `TakenCapture`). |
| level 18 | A capture's `frame_record` carries `held_layer`, the layer whose stored settings the camera held (None after a raw camera write), and its saved file `camera_held_a_layer` (and `camera_held_layer` when one was). A manual still has the camera hold its name's layer for its frame, and is refused while a manual recording runs when that would change the camera. |
| level 19 | The camera is its own object, `scope.camera` (`CameraAPI`; in-process also `session.camera`): every camera setter, read, hold, apply, save and restore moved there from `scope.imaging`, and the Session's `set_frame_size`, `set_binning_size`, `set_image_mode`, `set_high_conversion_gain`, `set_line_noise_reduction` and `frame_at_binning` with them, under the same names. Each storing setter saves what the camera took, whoever calls it, and `restore_camera_state` saves the binning, frame, unbinned region and image mode it puts back; the unsaved `imaging.set_frame_size`, `set_binning_size`, `set_pixel_format`, `set_conversion_gain_mode` and `set_line_noise_reduction` are gone, and a pixel format is chosen only through `set_image_mode`. `update_settings`' `has_member` refusal names the camera's members by their path, `scope.camera.<name>`. |
| level 20 | The camera's reads are the pair: a held property (`gain_db`, `exposure_ms`, `black_level`, `binning`, `frame_size`, `pixel_format`, `auto_gain`, `auto_exposure`, `high_conversion_gain`, `line_noise_reduction`), `None` whenever unknown (`active` a held `bool`), and a fresh `read_*` (`read_gain_db`, `read_exposure_ms`, `read_black_level`, `read_binning`, `read_frame_size`, `read_pixel_format`, `read_frame_rate_fps`) that raises `CameraReadFailedError` when the camera does not answer; the `get_*` getters, `get_live_camera_settings` and the `*_cached` properties are gone, and a frame whose exposure or gain cannot be read fails its capture. `set_binning` and `set_auto_exposure` replace `set_binning_size` and `set_auto_exposure_time`, and `apply` replaces `apply_layer_camera_settings`. |
| level 21 | `runtime_state.get_pixel_size_um()` and `runtime_state.get_field_of_view()`: the effective pixel size and the `FieldOfView(width_um, height_um)` at the current objective and the camera's held binning and frame, each `None` when it cannot be known; a plugin reads the scale here instead of computing it from an objective and a binning. |
| level 22 | Each setting the Session owns has its own read: `model`, `objective_id`, `turret_objective(position)`, `labware`, `scale_bar_enabled`, `acceleration_limit_pct`, `layer_acquire(layer)`, `layer_auto_gain(layer)` and `bookmark` (a `PlateLocation`), beside the existing `saved_focus(layer)` and the camera's reads; `runtime_state.get_turret_config()` and `imaging.scale_bar_config` are in-process. `camera.set_binning` and `camera.set_frame_size` answer the delivered `FrameSize`, not a dict. `get_setting` of a setting with its own read is refused `ArgumentRefusedError` (`has_member`), its `member` naming the read: `get_setting('scale_bar.enabled')` for `scale_bar_enabled`, `get_setting('image_mode')` for `scope.camera.image_mode`. |
| level 23 | Nothing a plugin outside Etaluma relies on. |
| level 24 | Each frame-validity source runs one of two rules, published as `capabilities.frame_rules` (a `FrameRule` per source: `rule` `Rule.CLOCK` or `Rule.CREDIT`, and under the clock rule its `delay_us` and `provenance`): the clock rule where the camera has a clock latch and a delay has been measured for it, the credit rule (the source's frame count) everywhere else. `imaging.judge_frame(frame, mapping)` takes the camera's clock mapping (a capture's `frame_record.clock_mapping`); a `Change` carries its `rule`, and a `Judgment` its `frame_start_ticks` and `qualifies_after_ticks` under the clock rule, in its `fields()` too. `imaging.recorded_changes()` answers each source's last `Change`. |
| level 25 | The diagnostics answers are records or published reads: `get_camera_link_info` answers a `CameraLink`, `get_motor_info` a `MotorBoardInfo` and `get_led_info` a `LedBoardInfo` (no `connected`: that is `led_connected`), each field `None` when unknown, an FX2 LED controller's and a missing motor board's firmware version included; `get_camera_temperatures_degc` is a `dict[str, float]`; `get_camera_diagnostic_info`, `get_camera_profile_info` and `get_system_info` are gone, their facts read from `capabilities` (`camera_model`, `camera_limits`, `pixel_size_um`, and new `camera_sensor`, `camera_shutter`, `camera_sdk_version`) and the camera's reads, the live exposure floor from `scope.camera.read_exposure_ms_min()`, which raises `CameraReadFailedError` when the camera does not answer. |
| level 26 | `session.controls_lock` answers a `ControlsLock` (`run_named`, `recording`, and `run_lockout` and `controls_locked` read from them): which of a person's controls the Session locks now, the one read every lock read reads; a plugin greys its own controls from it. |
| level 27 | The Session's ten run-state reads are gone: `is_protocol_running`, `run_in_progress`, `exclusive_activity`, `protocol_files_draining`, `protocol_files_pending`, `run_lockout`, `run_lockout_named`, `recording_active`, `controls_locked` and `motion_enabled`. `controls_lock` answers the lock, `live_work` what holds the scope and a finished run's files still to write, and a stage gesture reads `capabilities.has_xy_stage` beside the lock. |
| level 28 | A `Change` carries `earliest_ns`, the earliest moment it could have reached the hardware: a camera or LED write's start, or for a move once in force the start of the last read that saw its axis still moving (else its `written_ns`). A delay measured from it is never short; the clock rule still judges from `written_ns`, or a move's `in_force_ns`, so the pair never accepts a frame early. |
| level 29 | `session.plugin_run_in_flight()`: a context manager entered when a plugin's run starts and left when its worker ends, on any threads; while it is held, `live_work` lists a `'plugin_run'` item (`left` its count); it holds no claim and refuses nothing. |
| level 30 | On a camera with a clock latch (`capabilities.camera_has_clock_latch`), every move (`xy_move`, `z_move`, `turret`) runs the clock rule in `capabilities.frame_rules`: at its measured delay, or at 0 from the axis's stop when none has been measured, its `provenance` then `{'measured': 'none: a frame started after the stop'}`; a frame that started exposing before the stop never counts for the move. A camera reports a clock latch only when its frames carry their start (the `Timestamp` chunk); a Basler camera whose chunk did not turn on has no `camera_timestamp_tick_hz` and no latch. |
| level 31 | A move judged by the credit rule (`xy_move`, `z_move`, `turret` on a camera without a clock latch) counts its frames from its stop, not its drive: `motion_settled` places its `Change` in the frame stream there (`at_seq` the frames delivered at the stop, `credits` cleared, the source pending again), and no frame that arrived before it is credited. While the axis moves, `pending_sources` and `frames_until_valid()` answer the move's whole count, not 0 then 1. A later stamp of the axis settled starts the count again. |
| level 32 | A move is asked for by what it moves: `move_xy(x_um, y_um)` / `start_move_xy` and `move_xy_by(dx_um, dy_um)` / `start_move_xy_by` move X and Y as one request, every axis given refused before either drives, an axis given as None left as it is and not read; `move_z(z_um)` / `start_move_z` and `move_z_by(dz_um)` / `start_move_z_by` move Z, a move to a place approaching from below and a move by a distance not, unless `overshoot_enabled` (None, the API's approach, by default) says otherwise. `move_absolute`, `move_relative`, `start_move_absolute` and `start_move_relative` are gone, with no alias. |
| level 33 | A channel is a `channel` on the whole surface: `scope.channel_identity` (`ChannelIdentity`, its `channels` of `ChannelRecord`), `ScopeModel.channels`, `ShownChannel.channel` and `FrameRecord.held_channel`; `set_channel_acquire` / `channel_acquire`, `set_channel_auto_gain` / `channel_auto_gain` / `channel_auto_gain_lock`, `save_channel_focus`, `apply_focus_to_channel_steps` and `apply_channel_settings`; every member that takes a channel takes `channel` (`channels` on `add_step`); the reasons `channel_unknown` and `no_channel_selected`. The old names are gone, with no alias. |
| level 34 | A quantity's unit is the last part of its name: the camera's limits are `capabilities.camera_limits.gain_min_db`, `gain_max_db`, `analog_gain_max_db`, `exposure_min_ms` and `exposure_max_ms`; `scope.camera.read_frame_rate_fps()` and `read_exposure_min_ms()`. The old names are gone, with no alias. |
| level 35 | The image mode is a `lumascope.ImageMode` (`EIGHT_BIT`, `SCIENTIFIC`, `SCALED`, `FALSE_COLOR_RGB`): `scope.camera.set_image_mode` takes a member or its plain string and refuses any other value `not_offered` (`ArgumentRefusedError`) before anything is applied; `scope.camera.image_mode` answers a member. |
| level 36 | Nothing a plugin outside Etaluma relies on. |
| level 37 | An LED's board addresses are `led_addresses`: `scope.capabilities.led_addresses` (the installed board's, None without one) and a `ChannelRecord`'s `led_addresses` (the addresses the channel drives, `()` for none). |
| level 38 | An LED is lit by its channel's name: `scope.illumination.led_on(channel, illumination_ma)` and `led_off(channel)` take a name only, a number refused at the door (`wrong_argument_type`). `ch2color` and `color2ch` are not on the surface (no route); a channel's addresses are its record's `led_addresses`. `ScopeCapabilities.led_channels` and `ChannelRecord.led_channel` are gone, with no alias. |
| level 39 | A fluorescence channel is named for its excitation: the release's keys are `BF`, `PC`, `DF`, `Ex405`, `Ex488`, `Ex550`, `Ex589`, `Ex640` and `Lumi`, and a stock unit's are `Ex405` / `Ex488` / `Ex589` on LEDs 0 / 1 / 2, where `Blue` / `Green` / `Red` were. A fluorescence `ChannelRecord`'s `display_name` is `None` (its `label` is its excitation, `'405 nm'`). A caller's old name is refused, not translated: a channel `channel_unknown`, a settings path `not_a_setting`; a stored file's old names are read by the LED they were made with (see [Illumination](guide/light-and-camera.md#illumination) in the guide). |
| level 40 | Every steady stamp the scope takes is on one clock, the scope's, read by `session.host_steady_ns()` (in-process; `scope.host_steady_ns()`): a frame's `arrived_ns`, a `Change`'s `written_ns`, `earliest_ns` and `in_force_ns`, `Owed.until_ns`, a `ClockMapping`'s `host_ns`. It resolves to a microsecond or better on every supported Python and platform (`time.perf_counter_ns`), where on Windows under Python 3.12 the stamps were `time.monotonic_ns`, a 15.6 ms step. A plugin sets its own "now" against them by this member only, never by a clock of its own. |
| level 41 | Nothing a plugin outside Etaluma relies on. |
| level 42 | The Session's `camera`, `wellplate_loader` and `objective_helper` are gone: each was a second name for the scope's member of that name. The camera is `session.scope.camera`; the labware and objective catalogues are the scope's `wellplate_loader` and `objective_helper`. |
| level 43 | Every outcome class declares its reasons in `causes`, whatever its kind: a refusal's, a quiet outcome's and a raised notice's map each reason to its `RefusalCause`, a fault's and a reported notice's to None. A bare `cause` or `reason` in the class body is refused where the class is defined, and so is a class that carries a string another outcome class carries with another kind or cause, or without naming that class in `same_fact_as`; a subclass carries its parent's reasons as the same fact, and one that restates a parent's reason with another cause is refused. An outcome class is of one kind: a class under two kinds is refused, as is a class on the root that is no kind, and a subject below the umbrella carries no reasons. `RefusalCause` has a third value, `RESTART` (`'restart'`): no retry cures it until the server restarts (503 with no `Retry-After`). An outcome built with a reason its class does not declare raises `TypeError`, for every class. |
| level 44 | A position to keep is read by `scope.motion.positions_to_keep(axes, then=...)`: X, Y and Z in um from one snapshot of state and number, refused `axis_moving` while an asked axis moves, `position_unknown` while one homes or has lost its reference, and `axis_absent` for an axis the scope has no motor for, never answered 0.0. A plugin or script that keeps a position (a sweep's centre, a reference in a result, the Z it returns to) reads it here, never through `get_current_position`. `scope.protocols.focus_z_um` reads Z through it, and a bookmark and the session's `get_current_plate_position` read through it too. |
| level 45 | A capture's record comes back with its frame and nowhere else: `session.capture_still` (the scope's `imaging.capture_still` beneath it) returns a `CapturedStill`, its `record` the frame's `FrameRecord`, whose `exposure_source` / `gain_source` say whether each is the frame's own or read beside the grab and whose `auto_gain` is a locked arm's outcome. `imaging.last_capture_info` keeps only how the latest capture waited, frame or not (`hold_ms`, `drained`, `rechecks`, `deadline_s`, `n_entry`, `active_s`): its `frame_record`, `chunk_exposure_us`, `chunk_gain_db` and `auto_gain*` keys are gone. |
| level 46 | The turret's slot is read through `scope.motion.turret_status()` alone: `get_turret_slot`, `is_moving` and `is_current_slot_objective_set` are gone, and `get_axis_state` is no longer declared. On a scope with a turret (`capabilities.has_turret`) whose motor board is lost or never came up, `turret_status()` answers `TurretStatus('unknown', None)`, never None; None means the scope has no turret. `get_current_position` and `get_current_positions` read X, Y and Z only: `'T'` is refused `not_offered`, and the turret is left out of the positions. |
| level 47 | A record's time says what it is: `FrameRecord.captured_at` is a `FrameTime`, its `at` the time and its `source` the clock it came from (`TimestampSource`: `host_wallclock` for a still, read after the grab returned; `host_arrival` or `camera_rebased` for a recorded frame), so a time is never read without its source. `FrameRecord.illumination_timing` (`LightTiming`) says whether its `illumination_ma` was read before the grab or as the frame arrived. A record states an exposure or a gain only with its source (`exposure_source`, `gain_source` None exactly when the value is), and `judgments` is None for a frame that was never judged. A saved file's `Plane` names its `TimestampSource` and `IlluminationTiming`. |
| level 48 | The camera's firmware version is a capability: `capabilities.camera_firmware_version`, read once at connect beside `camera_serial_number`; None where the camera reports none (the FX2). |
| level 49 | One door per fact. The running model is `scope.channel_identity.model`: `ScopeCapabilities.model` is gone. Whether the scale bar is drawn is `session.scale_bar_enabled` and its colour `scope.imaging.scale_bar_color`, which `set_scale_bar_color` writes (over REST, `PUT scope/imaging/scale_bar_color`): `imaging.scale_bar_config` is gone. Gone with them, each for the read it derived: `motion.get_target_position` (`get_target_positions`), `motion.has_homed` and `position_is_known` (`axes_without_position`), `illumination.get_led_state` (`get_led_states`), `imaging.frame_is_valid` (`frames_until_valid() == 0`), `Protocol.step` (`step_records`), `Labware.get_dimensions` and `has_wells` (its fields), `BringUpRecord.part` and `substitution` (`parts`, `substitutions`). No longer declared, though still callable: the scope's `host_steady_ns` (`session.host_steady_ns`), `motor_connected`, `led_connected`, `camera_connected`, `camera.active` and `camera.camera_removed` (`unconnected_parts`), `runtime_state.get_labware` (`scope.labware[session.labware_id]`), `Protocol.steps` and `step_row`, and the post-processing `stitch`, `zproject`, `composite` and `video` (`build`). |
| level 50 | A camera snapshot is whole: `save_camera_state` holds the conversion-gain mode and line-noise reduction (`high_conversion_gain`, `line_noise_reduction`, the values the camera last took) with their saved settings in `stored`, and `restore_camera_state` puts both back where they differ, and saves them back. |
| level 51 | A post to the UI thread from another thread goes through `_lumascope.modules.kivy_utils`, one ordered queue: `schedule_ui(func)` runs an event once, in the order it was posted, and takes no delay; `schedule_ui_latest((owner, name), func)` keeps one entry per key, moved to the back with its newest function, so a held-up UI thread holds one entry per state and a state never runs ahead of an event posted before it. A raw `Clock` post joins no order with them. |
| level 52 | A camera write or read whose driver raised something other than a hardware fault (`HardwareError`) is our own defect and leaves as itself, never as `CameraSettingRejected` or `CameraReadFailedError`. `imaging.add_frame_listener` raises only `not_connected`: `FrameListenerNotRegisteredError` is gone, and with it `CaptureFailedError`'s `'recording_not_started'`, since registering a listener asks nothing of the camera. |
| level 53 | Our own defect is reported as LumaViewPro's, never as the hardware's: a protocol run ends at once as `run_loop_crashed` on an exception that is not a `LumascopeError`, in words naming LumaViewPro; a camera whose bring-up raised one is down with the cause `camera_bring_up_failed`. `ScopeDisconnectError` names `DisconnectPart.LEDS_OFF` when the LEDs' off at disconnect was not confirmed. |
| level 54 | A controller board answers what happened: a text command it did not answer raises `BoardNoReplyError`, a port that failed or is closed before the command was sent raises `BoardPortFailedError`, and one that failed after it was sent raises `BoardNoReplyError`, since the board may have taken it, where the driver's `exchange_command` and `exchange_multiline` answered None. The diagnostic reads of the motor board (`read_motor_drv_status`, `read_motor_fan_rpm`) raise them too. `send_diagnostic_command` and `send_diagnostic_command_multiline` refuse `'not_on_this_scope'`, `MissingPart.TEXT_COMMANDS`, for a board that takes no text commands (a TMCM-6110, a Classic's FX2 LED controller), and `led_on(block=True)` that the board answers but never confirms raises `BoardReplyUnreadableError`. |
| level 55 | A board's probes and reads answer what the board said: `read_led_currents_ma` raises `BoardNoReplyError`, `BoardPortFailedError` or `BoardReplyUnreadableError` for a channel the board did not report, where that channel read None; None now means only a board with no current sensing. `set_acceleration_limit` raises the board's `BoardNoReplyError` or `BoardPortFailedError` where a missed limit read answered a default. A motor controller that does not take the stored limit at bring-up is up with the cause `acceleration_not_set`. |
| level 56 | `autofocus_functions.autofocus_score(image)` is the score autofocus judges a frame by, its centre quarter's focus score, and autofocus scores through it. `focus_function` raises `ValueError` for a frame that is not 8-bit, since its saturation mask takes 255 as full scale; a frame from `capture_and_wait` or `get_image` with `force_to_8bit=True` is 8-bit. |
| level 57 | The scale bar's colour is the frame's: black where the frame inside the bar's box is bright, white where it is dark, for every capture whoever makes it. `ImagingAPI.scale_bar_color` and `set_scale_bar_color` are gone, with their route; `common_utils.get_bright_background_layers` is gone. `image_utils.add_scale_bar` takes `frames_summed` and `frame_bits` in place of `color` and `significant_bits`. |
| level 58 | `ScopeSession.set_scale_bar` raises `HardwareCommandRefusedError('exclusive_activity_running')` while a run, a diagnostic, a home or a recording holds the scope and the value is a change; the value already set is admitted. |
| level 59 | The plugin contract is declared: `session.plugins` is a read, and the namespaces' members this tutorial teaches are API. `PluginSpec.subscribes_to` and `ProcessorResult.metadata` are gone, and a record built with either raises `TypeError`. The mount point is a `UIMountPoint`: another name is refused `ArgumentRefusedError` (`not_offered`) where `ui.register` refused `PluginRegistrationError` (`unknown_mount_point`), and `UI_MOUNT_POINTS` is gone. A processor that is not callable is refused at `post_processing.register`, `ArgumentTypeRefusedError` (`wrong_argument_type`). |
| level 60 | One capture: `ImagingAPI.capture_and_wait` is gone. A caller below the Session takes `imaging.capture_still(...).image`, the same capture with the same arguments, its record beside the pixels (`CapturedStill.record`); `ScopeSession.capture_still` stays the Session's door. The capture's log lines say `capture_still:`. |
| level 61 | One remedy door: `ScopeSession.recover_file_writer` and `repair_camera_driver` are gone, with their routes. A remedy is taken through `session.apply_remedy(Remedy(...))`, which answers and refuses as each did, a held control session included. |
| level 62 | One log stamp: `_lumascope.modules.log_stamp` (standard library only, nothing at import) holds `make_formatter(layout)`, whose every line starts `2026-10-29 17:10:57.320`, `TRACE_LAYOUT` (serial, camera and api logs), `MAIN_LAYOUT`, `LOG_DATEFMT` and `now_stamp()`. Every LumaViewPro log line starts with that stamp. |
| level 63 | `ScopeSession.engineering_mode` is a read: storing to it raises `AttributeError`. `session.set_engineering_mode(enabled)` is its one writer, a guarded write that logs, refused `ControlSessionRefusedError` (`control_session_held`) while another holds the control session. |
| level 64 | Precision is Z's: `MotionAPI.set_precision_mode(enabled)` takes no axis and sets Z's. A call still passing `'Z'` first is refused `ArgumentTypeRefusedError` (`wrong_argument_type`) before anything is sent; over REST, a body still carrying `axis` is refused 422. `IlluminationAPI.save_led_state` and its record `LedSnapshot` are gone: read `get_led_states()` and hand what it answered to `restore_led_state(readings)`. `ScopeSession.create_protocol_runner()` is `protocol_runner()`, the same one runner, its route `POST /api/v1/protocol_runner`. |
| level 65 | A still's time is its first frame's start on the camera's clock: `FrameRecord.captured_at.source` is `TimestampSource.CAMERA_FRAME_START` (`camera_frame_start`), the frame's `Timestamp` tick mapped through the capture's `clock_mapping`, or `HOST_ARRIVAL` (`host_arrival`), the frame's arrival, on a camera without a latch or for a frame that came without its tick. A still no longer states `host_wallclock`, which stays the value `4.0.0-beta` files carry. For a summed still every fact of the record (time, `camera_timestamp_ticks`, `frame_id`, exposure, gain) is the first summed frame's. `ClockMapping` gains `wall_ns`, the host's wall clock at its `host_ns`, and `wall_time_at(ticks)`. |
| level 66 | A recording's time is each frame's start on the camera's clock: the recording takes one mapping as it starts (`ImagingAPI.map_camera_clock()`, new, on the camera lane; None on a camera without a latch) and maps every frame's `Timestamp` tick through it, so a recorded frame's `FrameRecord.captured_at.source` is `camera_frame_start`, or `host_arrival` on a camera without a latch. `camera_rebased` is gone. A frame that came without its tick in a recording whose times are the camera's is not kept: `RecordingResult.frames_unstamped` (and the manifest's `frames_unstamped`) counts it, `VideoFramesDroppedError` takes an `unstamped` count, and a recording whose every frame came that way ends as `RecordingFramesUnstampedError` (`frames_unstamped`). `RecordingResult.timestamp_grade` and the manifest's `timestamp_grade` are the recording's `TimestampSource` (`camera_frame_start` or `host_arrival`), no longer `camera`/`host`; `RecordingConfig` takes a required `timestamp_source`. |
| level 67 | One answer to whether a frame would count: `ImagingAPI.changes_in_force(exclude_sources=())`, now on the wire (`GET /api/v1/scope/imaging/changes_in_force`; its `mapping` is in-process only and defaults to None, answered without the clock mapping's uncertainty). It is the capture's own answer, so under the clock rule it waits a change's delay out. `ImagingAPI.frames_until_valid` and its route are gone, and so are `FrameValidity.is_valid`, `is_valid_for`, `frames_until_valid`, `pending_sources` and `reset()`: ask `changes_in_force()` for yes or no, `frame_validity.owed()` for what is still owed (`frames`, `until_ns`, and `moving`, new: the motion sources owed a frame because their axis reads moving), and `frame_validity.invalidation_counts`, a snapshot before and a compare after, for whether a change was recorded. `FrameValidity.unsettled_motion_sources` is gone: read `owed().moving`. |
| level 68 | A recording states its frame budget: `RecordingResult.frame_budget` and the manifest's `frame_budget`, `ceil(fps x duration)` with a rate limit and None without one. A camera slower than the limit ends at the duration as `duration_elapsed` with fewer frames than the budget, so `frames_selected` beside it says how much it delivered. `RecordingResult.summary()` is the one line each host logs as the recording finishes. |
| level 69 | `_lumascope.lib.log_helpers.VERBOSE` (15), a log level between DEBUG and INFO: the drivers' routine hardware traffic (a serial exchange, a camera setter, an LED lit, a move) is written at it, no longer INFO, and the `LVP.serial`, `LVP.camera` and `LVP.api` loggers pass it whatever the session's level, which sits on their files. A plugin that copies a trace takes it at VERBOSE. |
| level 70 | Counted frames are the one frame rule, on every camera and for every source, superseding the clock rule of levels 24, 28, 30 and 67: a change waits its source's count (a move's from its stop), and no frame is judged by when it started exposing. `Rule`, `at_the_stop`, `FrameRule.rule` and `FrameRule.delay_us` are gone: a `FrameRule` is its `count` and `provenance`. `imaging.judge_frame(frame)` and `frame_validity.judgments(frame)` take no clock mapping; `imaging.changes_in_force(exclude_sources=())` has no `mapping` argument; `frame_validity.owed(exclude_sources=())` takes no `uncertainty_ns`, and `Owed` has no `until_ns`; a `Judgment` has no `frame_start_ticks` or `qualifies_after_ticks`, and its `fields()` and a saved file's `frame_validity` no `rule`. `FrameValidity.rules_from_record(record)` takes the record alone, and a timing record whose entry holds an `apply_delay_us` is refused by that name. The clock latch stays: `capabilities.camera_has_clock_latch`, `imaging.map_camera_clock()`, a capture's `frame_record.clock_mapping` and a recording's frame times. |
| level 71 | The UI thread's one queue (level 51) is `_lumascope.modules.ui_queue`, and `_lumascope.modules.kivy_utils` is gone: every name it held is imported from `_lumascope.modules.ui_queue`, unchanged. The old name said Kivy; the module imports none, and the GUI host hands it the dispatcher. |
| level 72 | A plugin handler's caught exception is recorded as a `PluginRuntimeFailure` (it was `PluginRuntimeError`), kept in `NamespaceHealth.last_runtime_failures` (it was `last_runtime_errors`); its fields are unchanged. The record is never raised, and `except PluginRuntimeError:` raised `TypeError`. Over REST, the schema and the field carry the new names. |
| level 73 | LumaViewPro's code is one package, `_lumascope`, in the checkout's `src/`: every module keeps its name under it (`modules.x` is `_lumascope.modules.x`, `drivers.x` is `_lumascope.drivers.x`, `lvp_logger` is `_lumascope.lvp_logger`), and no top-level `modules`, `drivers`, `lib`, `rest` or `lvp_logger` remains. A plugin that imports an old name fails to import. |
| level 74 | A channel is called a channel on the surface: the refusal reasons `channel_not_on_scope`, `no_acquiring_channel`, `channels_unresolved`, `camera_holds_another_channel`, `composite_two_transmitted_channels` and `channel_setting_capped` (they were `layer_not_on_scope`, `no_acquiring_layer`, `layers_unresolved`, `camera_holds_another_layer`, `composite_two_transmitted_layers`, `layer_setting_capped`); the types `ChannelCameraRequest`, `ChannelHold` and `ChannelSettingCappedNotice` (they were `LayerCameraRequest`, `LayerHold`, `LayerSettingCappedNotice`); `capabilities.led_channels`, the channels that light an LED (it was `led_colors`; level 3's `led_channels`, the board addresses, has been `led_addresses` since level 37); a protocol build's `input_config['channel_configs']` (it was `'layer_configs'`); and the titles and words that said layer. No old name is kept. |
| level 75 | `NamespaceHealth.last_runtime_failures` is `runtime_failures`: it holds every handler failure since the namespace was created, never only the last ones. Over REST, the `plugin_health` answer carries the new name, and so does the support bundle's `plugins.json`. No old name is kept. |
| level 76 | A camera feature probe that fails as the camera comes up is published as failed, not as absent: `capabilities.camera_supports_conversion_gain_mode`, `camera_supports_line_noise_reduction`, `camera_supports_black_level`, `camera_has_clock_latch` and `camera_reports_temperature` are `bool | None`, `None` (REST `null`) meaning the probe failed and whether the camera has the feature is not known. Bring-up keeps the saved conversion gain and line-noise setting unapplied (the bring-up record's camera part says so); their setters and `set_black_level` probe the camera live on `None`, and a probe that fails there raises `CameraReadFailedError` where it refused as absent. |
| level 77 | `motion.home` and `start_home` take `put_back: bool | None`: with the stored setting `motion.restore_after_home` on (off as shipped), a home puts the turret back on its slot and Z back at its target; `None` follows the setting, `True` or `False` decides that home. A plugin that times or reads a home where it left the axes asks `put_back=False`. |
| level 78 | A camera's grab state that it does not answer is a failed read, not "not streaming": `scope.imaging.is_streaming()` and `is_streaming_once_settled()` raise `CameraReadFailedError`, and the session's `Status.camera_streaming` is `bool | None`, `None` (REST `null`) when the read failed, the failure logged at each ask. A camera declared removed still answers `False`. A write whose read fails before it is sent -- the feature probes `set_high_conversion_gain` and `set_line_noise_reduction` make, `set_black_level`'s probe and range read, `set_frame_size`'s binning read, `restore_camera_state`'s compare reads -- raises `CameraSettingRejected` reason `'camera_read_failed'`, nothing sent, where at level 76 it raised `CameraReadFailedError`. |
| level 79 | `motion.get_target_positions()` answers None (REST `null`) for an axis whose position is not known -- UNKNOWN or HOMING, or every axis while the motor board is lost -- as `axis_positions()` does; it answered the axis's last polled number, or 0.0 for one never read since bring-up. Every other answer is as before. |
| level 80 | A saved still's record is `SavedStill.record`, the name every other holder of a `FrameRecord` uses (`CapturedStill.record`); it was `SavedStill.frame`. Over REST, the still job's result carries the key `record`. No old name is kept. |
| level 81 | `diagnostics.send_diagnostic_command` and `send_diagnostic_command_multiline` answer with the board's reply or raise: a board that is not there is refused as every command for it is (`HardwareCommandRefusedError`, `'not_connected'`, or `'axis_absent'` with `MissingPart.MOTORS` for a motor command on a manual scope), a board that does not answer raises `BoardNoReplyError`, a failed port `BoardPortFailedError`, and an unknown target `ValueError`. They answered `'None'`, `'No response'`, `'Board not connected'` or `'Error: <msg>'` in place of a reply; those strings, and `is_board_reply`, are gone. A firmware `ERROR:` line is the board's reply, as before. `read_motor_drv_status` and `read_motor_fan_rpm` raise the same refusal for a board that is not there; None means only that the board's firmware does not offer the read. |
| level 82 | The frame-validity store is the API's own and no longer API: `scope.frame_validity`, `scope.camera.frame_validity` and `scope.imaging.frame_validity` are gone, superseding levels 14, 67 and 70 where they taught `frame_validity.count_frame`, `owed()`, `invalidation_counts` and `judgments(frame)`. Only the API records a change or counts a frame. Ask the imaging API: `changes_in_force(exclude_sources=())` whether a frame arriving now counts, `changes_in_force_limit_s(exclude_sources=())` how long the changes may take to come into force, `judge_frame(frame)` whether a frame counts for each change, and `recorded_changes()` each source's last change, for whether one was recorded. |
| level 83 | `session.settings` is read-only at every depth: a write into it, at a leaf, a block or a list, raises (`TypeError` or `AttributeError`), where before it reached the store past the range, the not-a-setting refusal and a held control session. Write through `session.update_settings(path, value)` or the setting's own Session member; read one setting with `session.get_setting(path)`; edit a copy with `copy.deepcopy(session.settings[...])`, a plain dict. |
| level 84 | LumaViewPro logs everything under one logger, `LVP`, the main log's own: the old second root, `lvp_logger`, is gone, and a record on it gets no session level and misses the main log below INFO. A plugin logs on `LVP` or a name under it (`LVP.<plugin>`); the REST log's logger is `LVP.rest`. Etaluma's engineering plugin follows from 1.0.206. |
| level 85 | A capture holds a channel by its name alone: `scope.imaging.capture_still(channel=None, ...)` takes the channel whose stored gain, exposure and auto-gain the camera holds for the grab (it took a `ChannelHold`), read from the store, and refuses a channel that is no channel (`channel_unknown`) or that this scope lacks (`not_on_this_scope`) before anything reaches the camera. `ChannelHold` and `ChannelCameraRequest` are gone: no member takes or answers them, and a caller can no longer have the camera hold settings its channel does not have stored. A positional `None`, and no channel, capture with the camera as set, as before. |
| level 86 | A run's blown, dark and unfocused captures are counted by step: `captures.saturated` and `captures.dark` are each a tuple of `StepFrames` (`step_index`, `step_name`, `count`, `first_scan_index`), and `focus_failures` holds one `FocusFailure` a step (`step_index`, `step_name`, `count`, `first_scan_index`, `cause`, the first failure's words), where each was one entry a capture or a scan. `TakenCapture` is gone. A run's held memory no longer grows with its scans. |
| level 87 | A scope's parts have one vocabulary. `Part` names every part a refusal names: the three boards, the motors, a stage axis's motor (`AXIS`), the turret, the XY stage, a channel and a channel's LED, an LED address, fan control, LED engineering mode and text commands. `Board` (`MOTOR`, `LED`, `CAMERA`, each also a `Part` by the same value) is what `PartStatus.part`, `unconnected_parts()` and the simulator's fault members take and answer, where they took `Part`. `MissingPart` is published and built one way, `MissingPart(Part.X, ...)`, carrying its `part` with the `axis`, `channel` or `address` that says which; its constants (`MissingPart.CAMERA` and the rest) and its `axis`, `channel`, `led` and `led_address` constructors are gone, `MissingPart.of_axis` taking an axis name. `HardwareCommandRefusedError.missing` is published, so the REST problem body names the part. The LED board is the LED controller everywhere: bring-up's progress reads `Connecting the LED Controller`, and `PartStatus.describe()` labels it `LED Controller`. |
| level 88 | A run names the boards it lacks as data. `ProtocolRunRefusedError` refused `'not_connected'` carries `unconnected_parts`, each a `Board`, and `RunOutcome.unconnected_parts` names the boards a `'hardware_disconnected'` ending lost; both are published, so the REST problem body and the `run_ended` event carry them. Every other refusal and ending carries none. |
| level 89 | A refusal or an ending names the axes it cannot place as data, in `axes_without_position`, each `Axis` mapped to its `AxisState`, as `MotionAPI.axes_without_position()` answers. `AxisStateUnknownError` carries it, renamed from `axes`, and its `axis` (the first of them) is gone; `ProtocolRunRefusedError` carries it for `'position_unknown'` and `'step_position_unknown'`, and `RunOutcome` for a `'position_lost'` ending. All three are published, so the REST problem body and the `run_ended` event carry the axes. Every other refusal and ending carries none. |
| level 90 | A disconnect that did not complete names each failed step on its board as data, in `ScopeDisconnectError.failures`: one `DisconnectFailure` per step, in teardown order, carrying the `Board`, the `DisconnectStep` (`leds_off` on the LED controller, `motor_stop` on the motor controller, `disconnect` on any board) and the step's reason. It replaces `parts` and `part_reasons`, and `DisconnectPart` is gone, so a board is spelled as everywhere else in the API (`'led'`, `'motor'`, `'camera'`). |
| level 91 | The installation's catalogues are looked up and listed through four members of the scope. `scope.plate(labware_id)` answers a plate's `Labware` under its catalogue name or a retired one, and `scope.objective(objective_id)` an objective's `Objective` by its catalogue key; each refuses a name its catalogue does not hold with `CatalogueNameRefusedError`, its `offered` the names it holds. `scope.all_labware` and `scope.all_objectives` list every record, in the order the catalogue offers them, each carrying the name its lookup takes; over REST each list is one `GET`. |
| level 92 | The catalogues are reached only through those four members. `scope.labware` and `scope.objectives`, the mappings by name, are no longer declared or served (`GET scope/labware`, `GET scope/objectives`): a plate is `scope.plate(labware_id)`, the selected plate's record `scope.plate(session.labware_id)` (level 49 named `scope.labware[session.labware_id]`), an objective `scope.objective(objective_id)`, and every record `scope.all_labware` or `scope.all_objectives`; the loaders level 42 names, `wellplate_loader` and `objective_helper`, are not declared either. `Protocol.validate_steps` is no longer declared: its argument was the scope's catalogue, and every caller is told of a protocol's invalid steps when it loads and is refused them at a run start. |
| level 93 | The camera's gain and exposure range is `scope.camera.setting_limits` (`CameraSettingLimits`: `gain_min_db`, `gain_max_db`, `analog_gain_max_db`, `exposure_min_ms`, `exposure_max_ms`), read from the camera at connect and after each pixel format, binning, frame size, conversion gain or line-noise change, so it is the range in force; a read that failed raises `CameraReadFailedError` from it until one succeeds. `capabilities.camera_limits` keeps only `frame_min` and `frame_max`, and `camera.read_exposure_min_ms` is gone: the floor is `setting_limits.exposure_min_ms`. The documented gain split is on capabilities, `camera_analog_gain_max_db` and `camera_gain_has_digital`, and `ScopeSession.channel_slider_maxima` answers the gain slider's maximum too, `slider_max_gain_db`. |
| level 94 | A display reads the optics as a state. `runtime_state.optics_status()` answers `OpticsStatus`: `objective`, the active objective's id or its `ObjectiveUnknownError`, and `field_of_view`, the `FieldOfView` or the first reason it is unknown (`ObjectiveUnknownError`, the camera's `HardwareCommandRefusedError`, `PixelSizeUnknownError`); a refusal is answered as the field's value, never raised. `runtime_state.frame_point_plate(x_fraction, y_fraction)` answers the plate position of a point of the frame, or the first reason it is unknown: the stage's `HardwareCommandRefusedError` (no motors, no X or Y), `AxisStateUnknownError` (X or Y not known), then the field of view's. |
| level 95 | The objective and the frame's scale answer their value or refuse with the first cause, never None. `session.objective_id` raises `ObjectiveUnknownError` with its reason (`'slot_unknown'`, `'slot_unassigned'`, `'not_in_catalogue'`, `'none_selected'`); `runtime_state.resolve_current_objective`, the second read of it, is no longer declared or served. `runtime_state.get_pixel_size_um`, `get_field_of_view` and `frame_point_distance` raise, in this order, `ObjectiveUnknownError`, the camera's `HardwareCommandRefusedError` (`'not_connected'`, `'scope_disconnected'`) and `PixelSizeUnknownError` (`'optics_undeclared'`, `'binning_unread'`, and for the two that need the frame `'frame_size_unread'`). A display reads both as a state through `optics_status` (level 94). |
| level 96 | Four names are no longer declared or served. `ManualRecordingController.discard_pending`: `session.discard_close_drain()` is the one way to drop a recording's unwritten frames. `scope.no_hardware`: `session.bring_up_record()` says which parts came up, each part's `up`. `Protocol.CURRENT_COLUMNS`, `CURRENT_VERSION` and `PROTOCOL_FILE_HEADER`, the protocol file's format, which a file states for itself. `lumascope.PLUGIN_API_LEVEL`: the level is `session.plugin_api_level`, which a plugin reads as 0 on a host older than it. |
| level 97 | `imaging.get_image` is an engineering member, and its `force_new_capture` is gone: it always waits for a frame the camera captures after the call. The last buffered frame is `imaging.get_image_from_buffer()`. |
| level 98 | `diagnostics.read_camera_node` is an engineering member, no longer served (`GET scope/diagnostics/read_camera_node`), and `capabilities.camera_node_maps`, the maps it reads, is no longer published: a node read by its name is for Etaluma's bench and support tools. A setting the API names has its own read, in the API's units (`camera.read_gain_db`, `read_exposure_ms` and the rest). |
| level 99 | Four calls that wait have a form that does not: `post_processing.start_build`, `post_processing.start_enhance`, `session.start_make_support_report` and `session.start_make_logs_zip`, each taking what its waiting form takes and returning an `Operation` done with what that form would have returned or raised. A REST client reads its job, whose `progress` is what the call last reported. |

---

## Common errors

### `PluginRegistrationError: Plugin 'foo' already registered in 'post_processing'`

Two plugins (or one plugin's `register` called twice) tried to claim
the same name in the same namespace. Pick a different `spec.name`,
or ensure your `register(ctx)` runs exactly once. Names are unique
**within a namespace**; the same name across `ui` and
`post_processing` is allowed but not recommended for clarity.

### `ArgumentRefusedError` (`not_offered`) from `ctx.plugins.ui.register`

The mount-point name you passed to `ctx.plugins.ui.register(spec, mount_point, builder)`
is not a `UIMountPoint`. The refusal's `offered` lists the known names;
in this release there is only `left_sidebar.accordion`. Mount points are
added deliberately as the host learns how to attach them; pick from
the listed set or ask for a new mount point with a widget-shape
contract.

### Startup notification: `The "foo" plugin did not load: version 0.1.0 needs LumaViewPro >=4.1.0, and this is 4.0.0`

Your `spec.requires_lvp_version` excludes the running LVP. Either
loosen the requirement (`>=4.0.0`), update LumaViewPro, or accept
that your plugin is gated to a future LVP. The host strips
pre-release suffixes (`-beta8`, `-rc1`) before comparing, so
`>=4.0.0` matches `4.0.0-beta8`.

### Startup notification: `The "foo" plugin did not load: its register() raised <ExceptionType>: <message>`

Your `register(ctx)` raised. The host removed whatever it had
registered before the exception -- nothing of the plugin is mounted,
run or called -- and then called your `unregister(ctx)` for any other
cleanup.
Read the log for the full traceback; common causes are typos in the
spec (e.g. `capabilities=("scope.imaging",)` vs `capabilities="scope.imaging"`), or
your processor closing over a missing module-level binding.

### `TypeError` when host invokes your processor

`ctx.plugins.post_processing` calls processors with three positional
arguments: `(input_dir: str, manifest: dict, output_dir: str)`. A
processor declared `def _process(self, ...)` (forgot to `@staticmethod`
the bound method) or with a different positional shape will raise at
invocation time, not at registration. Match the signature in the
hello-world example above.

### `ProcessorResult` constructor errors

`ProcessorResult` is a frozen dataclass. The most common slip is
passing a list where a tuple is declared:

```python
# Wrong -- a list does not raise here, but the record declares a tuple,
# and a frozen record holding a list is neither hashable nor frozen.
ProcessorResult(success=True, outputs=[summary_path])

# Right
ProcessorResult(success=True, outputs=(summary_path,))
```

Pass absolute paths in `outputs`; the host logs how many there are and
the folder they share, without normalizing them.

---

## Next steps

- For hardware-side context -- LED, motion, capture, protocol APIs
  your processor can read about (post-processors operate on saved
  files, not live hardware, but understanding what produced the
  files helps) -- see the guide, [`lumascope-skills.md`](lumascope-skills.md),
  and every member in [`api-reference.md`](api-reference.md).
- For per-frame listeners (running on the live stream, not after a
  run), see **Live frame listeners** above and
  [Live frames](guide/python.md#live-frames) in the guide.
- HTTP endpoints under `/plugins/<name>/` are not yet available; the
  namespace name is reserved.

If something this tutorial promised doesn't work, or if you hit a
case the API can't express, file an issue against
`EtalumaSupport/LumaViewPro` with the smallest reproducing plugin
package attached.
