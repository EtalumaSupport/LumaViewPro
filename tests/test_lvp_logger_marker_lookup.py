# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for the marker.lvpinstalled lookup path resolution.

Bug
---
The Runtime line in the startup banner reported "source / dev" on
installed MSI builds (visible in Chris's beta9 lumaviewpro.log:
"Runtime:   source / dev" despite the build being from the MSI
installer). Root cause: lvp_logger.py is bundled into the PyInstaller
exe via the .spec datas list, so under sys.frozen=True its __file__
points into the bundle extract dir (_MEI<random> in onefile mode,
<install>/_internal in onedir 6+) -- NOT the install root where the
WiX MSI drops marker.lvpinstalled. version.txt works because it's
bundled into the same dir via the .spec; the marker is intentionally
NOT bundled (it exists to distinguish "MSI-installed build" from
"PyInstaller dev build").

Fix
---
When sys.frozen is True, resolve the marker directory from
sys.executable's dirname instead of from __file__. The exe lives at
the install root, so its dirname is where the WiX MSI dropped the
marker.

Test approach
-------------
Source-level structural lock. lvp_logger runs the marker probe at
module import time, which makes behavioral testing via
importlib.reload heavy (would need to manipulate sys.frozen +
sys.executable across reloads and reset module-level state). The
structural test guards against the bug recurring by locking the
source-level invariants: (a) sys.frozen is checked, (b) sys.executable
is used when frozen, (c) the script_path fallback is used otherwise,
(d) the marker open() reads from the resolved dir, not the legacy
script_path-only path.
"""

from __future__ import annotations

import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
LVP_LOGGER_SRC = REPO / "lvp_logger.py"


class TestMarkerLookupResolvesFromExecutableWhenFrozen:
    """The marker.lvpinstalled probe must resolve its directory from
    sys.executable when sys.frozen is True. Without this, every MSI-
    installed user sees Runtime: source / dev in the log and the
    Documents-folder appdata path branch (lvp_appdata) wrongly resolves
    to script_path (a temp _MEI dir on onefile builds), making settings
    + logs non-persistent across runs."""

    def _source(self) -> str:
        return LVP_LOGGER_SRC.read_text()

    def _resolution_window(self) -> str:
        """Return the source window containing the marker-dir resolution +
        the probe open(). Skips past the explanatory comment block.

        Anchor on the `open(...marker.lvpinstalled...)` call line and walk
        backward ~25 lines so we span the resolution conditional AND the
        open() call itself. The comment block contains the string
        'marker.lvpinstalled' multiple times but no executable code, so
        anchoring on the open() is the unambiguous code-bearing anchor.
        """
        src = self._source()
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if "marker.lvpinstalled" in line and "open(" in line:
                window_start = max(0, i - 25)
                return "\n".join(lines[window_start:i + 2])
        raise AssertionError(
            "open() against marker.lvpinstalled not found in lvp_logger.py")

    def test_marker_lookup_checks_sys_frozen(self):
        """The marker probe must consult sys.frozen before resolving the
        directory. A future cleanup that drops the check reverts to the
        broken script_path-only behavior."""
        window = self._resolution_window()
        assert "sys.frozen" in window or "getattr(sys, 'frozen'" in window, (
            "marker.lvpinstalled probe must check sys.frozen before "
            "resolving its directory. Without this, frozen MSI builds "
            "report source / dev because __file__ points into the bundle "
            "extract dir. See class docstring."
        )

    def test_marker_lookup_uses_sys_executable_when_frozen(self):
        """The frozen branch must resolve from sys.executable's dirname,
        not from __file__ / script_path. The exe lives at the install
        root; the bundled .py does not."""
        window = self._resolution_window()
        assert "sys.executable" in window, (
            "marker.lvpinstalled probe must use sys.executable to resolve "
            "the install dir when frozen. Without this, the probe looks "
            "in _MEI<random> / _internal where the marker is not placed."
        )

    def test_marker_lookup_uses_script_path_fallback(self):
        """The non-frozen branch must still use script_path so dev runs
        from a source clone find the marker if a developer creates one
        for local testing."""
        window = self._resolution_window()
        assert "script_path" in window, (
            "marker.lvpinstalled probe must fall back to script_path "
            "when not frozen (dev source clone path)."
        )

    def test_marker_open_uses_resolved_dir_not_legacy_script_path(self):
        """The actual open() call must use the resolved marker_dir
        variable (not bare script_path). A future cleanup that
        accidentally re-uses script_path here re-introduces the bug
        for frozen builds."""
        src = self._source()
        # Find the open(...marker.lvpinstalled...) call and assert it
        # does NOT join against script_path directly. The resolved-dir
        # path uses os.path.join with our _marker_dir helper.
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if "marker.lvpinstalled" in line and "open(" in line:
                # Found the probe. Assert it does NOT use bare
                # script_path as the join base.
                assert "script_path" not in line, (
                    f"marker.lvpinstalled open() at line {i+1} uses bare "
                    f"script_path -- that's the legacy broken path. Use "
                    f"the frozen-aware _marker_dir variable instead. Line: "
                    f"{line!r}"
                )
                return
        raise AssertionError(
            "open() call against marker.lvpinstalled not found "
            "in lvp_logger.py")
