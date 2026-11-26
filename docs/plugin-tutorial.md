# LumaViewPro -- Plugin Tutorial

## PRE-RELEASE API

The plugin platform documented in this file is the 4.x plugin
contract: what this tutorial teaches freezes with the API at `4.0.0`, and
is pre-release until then, as the guide's preface says
([`api-guide.md`](api-guide.md)).
`ctx.plugins.post_processing` operates on saved files. `ctx.plugins.ui`
has one mount point, `left_sidebar.accordion`.
`ctx.plugins.live_processing` is a per-frame listener on the live camera
stream. `ctx.plugins.rest` is reserved and outside the freeze
([The reserved namespace](#the-reserved-namespace)).

---

## What a plugin is

A LumaViewPro plugin is a Python package that the host discovers at
startup via the `lvp.plugins` entry-points group. Each plugin exposes
a module-level `PluginSpec`, a `register(ctx)` function, and an
optional `unregister(ctx)`. `register(ctx)` calls one or more
`ctx.plugins.<namespace>.register(...)` methods to attach the
plugin's handlers; the host wraps the call in try/except so a bad
plugin cannot crash the app. `ctx` is the `ScopeSession` hosting the
plugin: the same session LumaViewPro's window, REST and scripts drive. A
plugin starts from it as every caller does: the session's members for what is the
session's (its settings, the shown channel, runs, captures), `ctx.scope.<api>`
for the hardware (`docs/api-guide.md`).

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
    files, started threads, or added listeners of their own (a run-state
    listener, a frame listener added through scope.imaging) must release
    them here so a clean re-load works; a frame listener registered
    through ctx.plugins.live_processing the host removes first.
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
against the running LumaViewPro version, then calls `register(ctx)`. If any
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

Run them with the Python LumaViewPro's install script set up
([Where it runs](guide/python.md#where-it-runs)), which imports `lumascope`
from any folder.

---

## Lifecycle and namespaces

### Load

A host asks its session for plugins with `session.load_plugins()`;
LumaViewPro does at startup. A script's or a test's session that never
asks has none. For each entry point in `lvp.plugins`, and then for each
built-in:

1. Import the module. Import failure -> not loaded.
2. Read module-level `spec`. Missing or not a `PluginSpec` -> not loaded.
3. Check the spec: a `requires_lvp_version` the running LumaViewPro version
   does not meet -> not loaded.
4. Call `register(ctx)` with the session. Exception -> remove whatever
   it registered before the exception (a widget, a processor, a frame
   listener), then call `unregister(ctx)`, if the plugin has one, for
   its own cleanup -> not loaded.
5. Track the module so `session.unload_plugins()` can call its
   `unregister` at shutdown.

A plugin that did not load is recorded with its reason, for
`session.plugin_health` and the tech-support report.

An installed plugin claiming a built-in's name keeps it; the built-in
is the one reported as not loaded.

Each step's failure leaves the rest of the app intact. The one report
when discovery ends says, for each plugin, `The "<name>" plugin did not
load: <reason>.`, then `The rest of LumaViewPro is unaffected.` -- if
you see that on startup, that's why.

### Unload

At app shutdown `session.unload_plugins()` walks the loaded list in
reverse, removes each plugin's frame listener from the camera, and
calls its `unregister(ctx)`; a session's
`shutdown()` does it first for a host that did not. Exceptions are
caught and logged at WARNING; shutdown is not blocked by a plugin's
teardown failure. It runs before the hardware is disconnected: a plugin
driving the scope stops that work here and waits, bounded, for it to
put the scope back.

Reverse order matters when plugin B was registered after plugin A
and depends on resources A exposes -- B comes down first.

### Settings change

No plugin is notified of a settings change (plugin API level 9): the session
saves every accepted write itself, and `on_settings_changed` is not
called. A plugin that needs a setting reads it when it needs it
(`ctx.setting('video.max_fps')`).

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
holds the scope. A plugin grays its own controls from
`ctx.add_status_listener` and `ctx.controls_lock.controls_locked`,
as the built-in regions do, so a control that must stay live during a
run, such as its own Stop, stays live.

### Live frame listeners

`ctx.plugins.live_processing.register(spec, handler)` adds `handler`
as a listener on every frame the camera delivers, live view and runs
alike. The registry forwards to `scope.imaging.add_frame_listener`
under your plugin's name, so the log names your plugin when the
handler runs over its budget or raises. The host removes it when it
unloads your plugin, before it calls your `unregister(ctx)`, so your
`unregister` does not remove it. This is the door for a listener that
lives as long as your plugin; a listener for a single action, one
characterization or one measurement, is added with
`scope.imaging.add_frame_listener(handler, name=...)` and removed by
whoever added it, with `remove_frame_listener` in a `finally`.

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
this release, so a shipped plugin registers none. Registering a route
there raises `PluginRegistrationError` (`rest_namespace_reserved`). The
namespace is outside the 4.0 freeze: a route would take a framework
router, which has no published type, and what a plugin route looks like
is not settled. Don't catch and ignore the raise; let your
plugin fail to load, and serve what it needs through `post_processing`,
`live_processing` or `ui`. While a plugin is being developed,
`ctx.plugins.rest.enable_stub_mode()` records a registration and serves
nothing, so the rest of the plugin can be tried.

---

## The doors a plugin uses

What a plugin calls today, whatever a row of
[`plugin-api-levels.md`](plugin-api-levels.md) once taught. Every
name here is published ([the reference](api-reference.md)); anything else
is not API and may change in any release. A plugin
starts from the session, `ctx`, as every caller does.

- **The session's own**: a setting (`ctx.setting(path)`,
  `ctx.update_settings(path, value)`), the shown channel
  (`ctx.show_channel`, `ctx.light_shown_channel`), a still
  (`ctx.capture_still`), a saved still and a recording
  (`ctx.manual_capture`, `ctx.manual_recording`), protocols and their runs
  (`ctx.create_empty_protocol`, `ctx.load_protocol`, `ctx.protocol_runner()`),
  the scope's clock (`ctx.host_steady_ns()`) and the level
  (`ctx.plugin_api_level`).
- **The hardware**, `ctx.scope.<api>`: `motion` (a position to keep,
  `positions_to_keep`; the axes, `axis_positions`), `illumination`, `camera`,
  `imaging` (whether a frame would count, `changes_in_force`,
  `judge_frame`), and the catalogs (`ctx.scope.labware`, `objective`,
  `all_labware`, `all_objectives`).
- **Registering**: `ctx.plugins.post_processing.register`,
  `ctx.plugins.live_processing.register` and `ctx.plugins.ui.register`;
  `ctx.plugins.rest` is reserved.
- **Listening**: a frame listener for the plugin's life through
  `ctx.plugins.live_processing.register`, and one for a single action
  through `scope.imaging.add_frame_listener`; a position, turret or LED
  listener through `scope.motion.add_position_listener`,
  `add_turret_listener` and `scope.illumination.add_led_listener`. Each is
  called on the thread that caused the change, so UI work goes to the UI
  thread.
- **Logging**: the `LVP` logger, or a name under it (`LVP.<plugin>`).

---

## The plugin API level

`session.plugin_api_level` says what
this LumaViewPro does that a plugin may rely on, within its major version.
A version cannot say it: many builds share one beta number. A change to
behavior a plugin relies on raises the level by one. A plugin that needs a level checks
it at its own entry and refuses below it, naming both levels; a host older
than the level has no `plugin_api_level`, which reads as 0
(`getattr(session, 'plugin_api_level', 0)`). `PluginSpec.requires_lvp_version`
still states the major version.

What each level added is a row of
[`plugin-api-levels.md`](plugin-api-levels.md).

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

Your `spec.requires_lvp_version` excludes the running LumaViewPro. Either
loosen the requirement (`>=4.0.0`), update LumaViewPro, or accept
that your plugin is gated to a future LumaViewPro. The host strips
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
  files helps) -- see the guide, [`api-guide.md`](api-guide.md),
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
