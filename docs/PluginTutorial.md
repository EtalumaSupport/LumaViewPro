# LumaViewPro -- Plugin Tutorial (Post-Processing, Phase A)

## PRE-RELEASE API

The plugin platform documented in this file is **Phase A** of the 4.x
plugin contract. The `ctx.plugins.post_processing` namespace shape is
stable. `ctx.plugins.ui` is registerable now but the only mount point
is `left_sidebar.accordion`. `ctx.plugins.live_processing` and
`ctx.plugins.rest` are name-reserved but raise on registration in
Phase A -- read **Lifecycle and namespaces** below before designing
around them.

If you are shipping a plugin against this contract before 4.x freezes,
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
plugin cannot crash the app.

Four namespaces are defined for 4.x:

| Namespace | Purpose | State in Phase A |
|---|---|---|
| `ui` | Adds widgets at named mount points | live; one mount point (`left_sidebar.accordion`) |
| `post_processing` | Operates on saved capture folders | live; intern's primary surface |
| `live_processing` | Per-frame listeners during capture | reserved; raises until Wave 7 Phase 4 |
| `rest` | HTTP sub-routers mounted under `/plugins/<name>/` | reserved; raises until REST design lands |

This tutorial covers `post_processing` end-to-end. UI / live / REST
shapes are documented elsewhere as those surfaces land.

---

## Hello-world post-processor

The smallest possible plugin: a processor that walks an input
directory, tags every TIFF it finds, and writes a one-line summary
into the output directory. No image edits; the goal is to see the
contract end-to-end.

### Directory layout

```
hello_postproc/
    pyproject.toml
    hello_postproc/
        __init__.py
```

### `hello_postproc/__init__.py`

```python
"""Hello-world post-processor for LumaViewPro.

Registers a processor that lists the TIFFs in input_dir and writes a
summary.txt into output_dir. Demonstrates the Phase A contract; not a
useful processor on its own.
"""
from __future__ import annotations

import os
from typing import Any

from modules.plugins import PluginSpec, ProcessorResult


__version__ = "0.1.0"


spec = PluginSpec(
    name="hello_postproc",
    version=__version__,
    requires_lvp_version=">=4.0.0",
    description="Lists TIFFs in a capture folder and writes a summary.",
    capabilities=("modules.image_save",),
    subscribes_to=(),
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
        metadata={"tiff_count": len(tiffs)},
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
| `description` | yes | One sentence shown in the plugin list and the tech-support report. |
| `capabilities` | no | Tuple of dotted paths the plugin uses (`scope.imaging`, `modules.image_save`). Recorded in tech-support reports; not sandbox-enforced in 4.x. |
| `subscribes_to` | no | Tuple of settings-tree keys (`video.max_fps`). If non-empty, the host will call your `on_settings_changed(ctx, settings)` when one of those keys changes. Empty tuple = hook never fires. |
| `author` | no | Free-form. |
| `url` | no | Free-form. |

`PluginSpec` is a frozen dataclass; attempting to mutate a field
after construction raises.

### What `ProcessorResult` means

| Field | Meaning |
|---|---|
| `success` | True if the processor completed. False puts the plugin on the failure list in the run-complete dialog. |
| `outputs` | Tuple of absolute paths the host should surface (e.g. "Open output folder", attach to email). |
| `message` | One-line user-facing summary. Speak to an L1 researcher; no internal IDs or exception class names. |
| `metadata` | Free-form dict. Recorded for the tech-support report; not surfaced in the UI. |

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
requires-python = ">=3.11"
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

```
pip install -e /path/to/hello_postproc
```

Restart LumaViewPro. At startup the host iterates entry points,
imports each one, reads its `spec`, checks `requires_lvp_version`
against the running LVP version, then calls `register(ctx)`. If any
step fails the plugin is skipped, a `notifications.error` fires, and
the rest of the app keeps loading.

---

## Testing your plugin

Use the `harness_ctx` fixture in `tests/plugin_test_harness.py`. It
builds a fresh `ctx` with a real `PluginRegistry` and mocked scope /
session / lumaview attributes, so your `register(ctx)` runs against
the production registry code without spinning up Kivy or hardware.

Place this in `tests/test_hello_postproc.py`:

```python
"""Tests for hello_postproc. Run with: pytest tests/test_hello_postproc.py"""
from __future__ import annotations

import os
import tempfile

from tests.plugin_test_harness import harness_ctx  # noqa: F401

import hello_postproc


def test_registers_in_post_processing_namespace(harness_ctx):
    hello_postproc.register(harness_ctx)
    assert "hello_postproc" in harness_ctx.plugins.post_processing.names()


def test_processor_writes_summary(harness_ctx, tmp_path):
    hello_postproc.register(harness_ctx)
    processor = harness_ctx.plugins.post_processing.get("hello_postproc")

    input_dir = tmp_path / "capture"
    input_dir.mkdir()
    (input_dir / "frame_001.tiff").write_bytes(b"")
    (input_dir / "frame_002.tif").write_bytes(b"")
    (input_dir / "notes.txt").write_bytes(b"")

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    result = processor(str(input_dir), {}, str(output_dir))

    assert result.success is True
    assert result.metadata["tiff_count"] == 2
    summary = (output_dir / "summary.txt").read_text(encoding="utf-8")
    assert "tiff_count: 2" in summary


def test_unregister_does_not_raise(harness_ctx):
    hello_postproc.register(harness_ctx)
    hello_postproc.unregister(harness_ctx)
```

The fixture gives you a fresh `ctx` per test; registrations do not
leak across tests. If you need multiple isolated contexts in one
test (e.g. to assert two contexts don't share state), use the
`harness_ctx_factory` fixture instead.

The harness re-exports `PluginSpec`, `ProcessorResult`, and
`PluginRegistrationError` so test files can construct specs without
importing from `modules.plugins` directly:

```python
from tests.plugin_test_harness import PluginSpec, ProcessorResult
```

---

## Lifecycle and namespaces

### Load

At app startup, after `AppContext` and the widget tree exist, the
host calls `load_plugins(ctx)`. For each entry point in
`lvp.plugins`:

1. Import the module. Import failure -> log + notify + skip.
2. Read module-level `spec`. Missing or wrong type -> warn + skip.
3. Check `requires_lvp_version` against the running LVP version.
   Incompatible -> warn + record on the failed-list + skip.
4. Call `register(ctx)`. Exception -> log + notify + call
   `unregister(ctx)` for cleanup + skip.
5. Track the module so `unload_plugins(ctx)` can call its
   `unregister` at shutdown.

Each step's failure leaves the rest of the app intact. The
`notifications.error` for a load failure says "Plugin X did not
load: <reason>. Other features unaffected." -- if you see that on
startup, that's why.

### Unload

At app shutdown `unload_plugins(ctx)` walks the loaded list in
reverse and calls each plugin's `unregister(ctx)`. Exceptions are
caught and logged at WARNING; shutdown is not blocked by a plugin's
teardown failure.

Reverse order matters when plugin B was registered after plugin A
and depends on resources A exposes -- B comes down first.

### Settings change (optional)

If your spec sets `subscribes_to=("some.setting.key", ...)`, the
host calls your module-level `on_settings_changed(ctx, settings)`
when one of those keys changes. Empty tuple (the default) means the
hook never fires; you don't need to define the function.

### Reserved namespaces

Calling `ctx.plugins.live_processing.register(...)` or
`ctx.plugins.rest.register(...)` in Phase A raises
`PluginRegistrationError` with a message that names the wave / phase
that will unlock the surface. Don't catch and ignore the raise; let
your plugin fail to load and rework the design against
`post_processing` until the target surface ships.

---

## Common errors

### `PluginRegistrationError: Plugin 'foo' already registered in 'post_processing'`

Two plugins (or one plugin's `register` called twice) tried to claim
the same name in the same namespace. Pick a different `spec.name`,
or ensure your `register(ctx)` runs exactly once. Names are unique
**within a namespace**; the same name across `ui` and
`post_processing` is allowed but not recommended for clarity.

### `PluginRegistrationError: Unknown UI mount point 'X'. Known: [...]`

The mount-point name you passed to `ctx.plugins.ui.register(spec, mount_point, builder)`
is not in `UI_MOUNT_POINTS`. The error message lists the known names;
in Phase A there is only `left_sidebar.accordion`. Mount points are
added deliberately as the host learns how to attach them; pick from
the listed set or ask for a new mount point with a widget-shape
contract.

### Startup notification: `Plugin X did not load: requires >=4.1.0, have 4.0.0`

Your `spec.requires_lvp_version` excludes the running LVP. Either
loosen the requirement (`>=4.0.0`), update LumaViewPro, or accept
that your plugin is gated to a future LVP. The host strips
pre-release suffixes (`-beta8`, `-rc1`) before comparing, so
`>=4.0.0` matches `4.0.0-beta8`.

### Startup notification: `Plugin X did not load: <ExceptionType>: <message>`

Your `register(ctx)` raised. The host already called your
`unregister(ctx)` for cleanup, so partial state should be released.
Read the log for the full traceback; common causes are typos in the
spec (e.g. `subscribes_to=("key",)` vs `subscribes_to="key"`), or
your processor closing over a missing module-level binding.

### `TypeError` when host invokes your processor

`ctx.plugins.post_processing` calls processors with three positional
arguments: `(input_dir: str, manifest: dict, output_dir: str)`. A
processor declared `def _process(self, ...)` (forgot to `@staticmethod`
the bound method) or with a different positional shape will raise at
invocation time, not at registration. Match the signature in the
hello-world example above.

### `ProcessorResult` constructor errors

`ProcessorResult` is frozen and positional. The most common slip is
passing a list where a tuple is required:

```python
# Wrong -- list will not raise here, but the host treats outputs as
# a sequence of strings, and downstream code may rely on tuple-shape
# (e.g. hashability of the result for diagnostic dedup).
ProcessorResult(success=True, outputs=[summary_path])

# Right
ProcessorResult(success=True, outputs=(summary_path,))
```

Pass absolute paths in `outputs`; the host surfaces them in the
run-complete dialog without normalization.

---

## Next steps

- For hardware-side context -- LED, motion, capture, protocol APIs
  your processor can read about (post-processors operate on saved
  files, not live hardware, but understanding what produced the
  files helps) -- see `LumascopeSkills.md` in this directory.
- For per-frame listeners (running during capture, not after) wait
  for `ctx.plugins.live_processing` to ship; its registration shape
  is locked.
- For HTTP endpoints under `/plugins/<name>/` wait for the REST
  design session to land; the namespace name is reserved.
- The full design rationale -- why four namespaces, why
  entry-points discovery, why no sandbox in 4.x -- is documented
  elsewhere and will be linked when it lands in this repo.

If something this tutorial promised doesn't work, or if you hit a
case the API can't express, file an issue against
`EtalumaSupport/LumaViewPro` with the smallest reproducing plugin
package attached.
