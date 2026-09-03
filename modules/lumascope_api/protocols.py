# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""ProtocolsAPI -- protocol-author surface: construct Protocol objects.

The protocol constructors need one thing the Protocol data class cannot
resolve for itself: where `data/tiling.json` lives. That path is a
property of the running INSTALLATION, not of any protocol, so the
constructors resolve it here and callers never pass
`tiling_configs_file_loc` by hand.

`source_path` arrives after construction rather than as a constructor
argument: the application builds the scope before it knows its own data
root. That ordering is why `register_source_path` is a separate call,
and why the constructors raise instead of guessing when it was never
made -- a wrong tiling config silently produces a protocol whose tiling
geometry does not match the instrument.

This surface BUILDS protocols; it does not run them. The runner is
`ScopeSession.create_protocol_runner()`.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope
    from modules.protocol import Protocol


class ProtocolsAPI:
    """Protocol-construction sub-API on a Lumascope.

    Hosts the two public constructors (`load_protocol`, `create_protocol`)
    and the installation data root both resolve against.
    """

    def __init__(self, scope: Lumascope) -> None:
        self._scope = scope
        self._source_path = None

    def register_source_path(self, source_path) -> None:
        """Register the LVP source/data path the constructors resolve against.

        Internal session-composition wiring -- called by ScopeSession at
        construction and not part of the L2 API surface.

        Called once at startup, after the scope is constructed. Tests that
        don't drive the protocol API can skip it.

        Args:
            source_path: Path-like to the LVP source/data root.
        """
        self._source_path = source_path

    def tiling_configs_path(self) -> pathlib.Path:
        """Resolve data/tiling.json from the registered source path.

        The one owner of that path: the protocol constructors resolve it
        here, and the run engine takes it from here at run start for the
        post-run composite merge and hyperstack build, so a headless run
        reads the same tiling config the session was built with instead
        of whatever the process's script root holds.
        """

        if self._source_path is None:
            raise RuntimeError(
                'scope.protocols.load_protocol/create_protocol require '
                'scope.protocols.register_source_path() to have been called.'
            )
        return pathlib.Path(self._source_path) / 'data' / 'tiling.json'

    def load_protocol(self, file_path: str | pathlib.Path) -> Protocol:
        """Load a Protocol from disk.

        Wraps ``Protocol.from_file(...)`` and resolves
        ``data/tiling.json`` from the registered source_path.

        Args:
            file_path: Path to the protocol file.

        Returns:
            Protocol: The loaded Protocol instance.

        Raises:
            ProtocolFormatError: On format issues (same surface as
                Protocol.from_file).
        """
        from modules.protocol import Protocol

        return Protocol.from_file(
            file_path=file_path,
            tiling_configs_file_loc=self.tiling_configs_path(),
        )

    def create_protocol(
        self,
        *,
        config: dict | None = None,
        input_config: dict | None = None,
        empty_config: dict | None = None,
    ) -> Protocol:
        """Construct a Protocol in-memory.

        Three modes (pass exactly one):
          - config={...}: full config dict passed to Protocol() directly.
          - input_config={...}: partial config (positions, layer_configs,
            etc.); routed through Protocol.from_config which fills defaults.
          - empty_config={...}: labware/objective config for an empty-steps
            protocol; routed through Protocol.create_empty.
        tiling_configs_file_loc is resolved internally from the registered
        source_path.

        Args:
            config: Full config dict, or None.
            input_config: Partial config dict, or None.
            empty_config: Empty-steps config dict, or None.

        Returns:
            Protocol: Newly constructed Protocol instance.

        Raises:
            ValueError: If exactly one of config/input_config/empty_config
                was not provided.
        """
        from modules.protocol import Protocol

        provided = sum(1 for x in (config, input_config, empty_config) if x is not None)
        if provided != 1:
            raise ValueError(
                'create_protocol(): pass exactly one of config=, input_config=, or empty_config='
            )
        tcfg = self.tiling_configs_path()
        if input_config is not None:
            return Protocol.from_config(
                input_config=input_config,
                tiling_configs_file_loc=tcfg,
            )
        if empty_config is not None:
            return Protocol.create_empty(
                config=empty_config,
                tiling_configs_file_loc=tcfg,
            )
        return Protocol(
            tiling_configs_file_loc=tcfg,
            config=config,
        )
