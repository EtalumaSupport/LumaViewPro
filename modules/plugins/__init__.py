# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Plugin platform for LumaViewPro.

Single platform, namespace-scoped registries, lifecycle-aware.
For plugin authors: docs/plugin-tutorial.md

Four namespaces hardcoded for 4.x: ui, post_processing, live_processing,
rest. Adding a fifth is a deliberate platform-spec change, not a runtime
extension. Decision held at four because the surfaces map to where
LumaViewPro can be extended: UI tree, batch processing of saved
captures, per-frame processing during capture, and external HTTP
clients. New extension surfaces should be considered against those
axes before a new namespace is added.

Plugin authors implement:
    __version__ = "X.Y.Z"
    spec = PluginSpec(...)
    def register(ctx): ...
    def unregister(ctx): ...                    # optional

``ctx`` is the ScopeSession hosting the plugin. A host asks its session
for plugins (``session.load_plugins()``); the session discovers them via
entry_points group 'lvp.plugins'.
"""

from __future__ import annotations

import enum
import importlib.metadata
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

from modules.exceptions import (
    PluginFailedError,
    PluginNotLoadedError,
    PluginProcessorSkippedError,
    Refusal,
    RefusalCause,
)
from drivers.camera import Frame
from modules.notification_center import notifications
from modules.api_surface import FilePath, api, api_fields
from modules.run_outcome import FilesOutcome

logger = logging.getLogger('lvp_logger')


ENTRY_POINT_GROUP = 'lvp.plugins'

# The oldest version of each known plugin this LumaViewPro works with.
# A plugin's requires_lvp_version only guards the other direction, so a
# plugin older than a host change it reaches would otherwise load and
# fail mid-use. A host change the plugin must follow raises the value
# here, in the same commit, to the plugin version that follows it.
MINIMUM_PLUGIN_VERSIONS = {
    'etaluma_engineering': '1.0.158',
}

# Etaluma's bench tool. Once installed it cannot be disabled any other way,
# so a host started with no_engineering leaves it out before it is imported:
# the machine behaves as if it were not installed.
ENGINEERING_PLUGIN = 'etaluma_engineering'

# What this LumaViewPro does that a plugin may rely on, within its major
# version: no version can say it, since version.txt names the promoted
# release and many trunk commits share one beta number. A commit that
# changes behaviour a plugin relies on raises it by one, in that commit,
# with its row in docs/plugin-tutorial.md ("The plugin API level"). A plugin
# reads it at its own entry through session.plugin_api_level and refuses
# below what it needs; a host older than the level has no such member,
# which the plugin reads as 0.
PLUGIN_API_LEVEL = 59


class UIMountPoint(enum.StrEnum):
    """Where the host attaches a plugin's widget: the mounts it knows how to attach.

    A mount is added when a real consumer needs one, with a widget-shape
    contract for that mount.

    Attributes:
        LEFT_SIDEBAR_ACCORDION: An item of the left sidebar's accordion; the
            builder answers a Kivy ``AccordionItem``.
    """

    LEFT_SIDEBAR_ACCORDION = 'left_sidebar.accordion'


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PluginRegistrationError(Refusal, Exception):
    """A plugin asked a namespace for something this LumaViewPro does not give.

    The plugin is NOT loaded and the app continues. The loader catches the
    raise from the plugin's register(ctx) and reports it as that plugin's
    load failure. The same registration fails again unchanged, so its cause
    is the request.

    Attributes:
        reason: ``'name_taken'`` -- the namespace already holds a plugin of
            that name; ``'rest_namespace_reserved'`` -- a plugin cannot add
            REST routes in this release.
    """

    title = 'Plugin Not Registered'
    causes = dict.fromkeys(('name_taken', 'rest_namespace_reserved'), RefusalCause.REQUEST)

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@api_fields(
    'name',
    'version',
    'requires_lvp_version',
    'description',
    'capabilities',
    'author',
    'url',
    'auto_run_on_protocol_complete',
)
@dataclass(frozen=True)
class PluginSpec:
    """Declarative metadata a plugin presents at registration time.

    Attributes:
        name: The plugin's name, unique within a namespace.
        version: The plugin's version.
        requires_lvp_version: The LumaViewPro versions it runs on, as a
            requirement (``'>=4.0.0'``); pre-release suffixes are stripped
            before comparing.
        description: One sentence saying what the plugin does.
        capabilities: The API surfaces it uses, as dotted paths
            (``'scope.imaging'``). A declaration, not a sandbox: the
            tech-support report and diagnostic probes record it.
        author: Its author.
        url: Its web address.
        auto_run_on_protocol_complete: For a post-processing plugin, True
            hands its processor the folder of every Full Protocol run,
            whoever started it, once its files are on disk and its
            hyperstack build has ended
            (``PluginRegistry.run_protocol_complete_processors``).
    """

    name: str
    version: str
    requires_lvp_version: str
    description: str
    capabilities: tuple[str, ...] = ()
    author: str = ''
    url: str = ''
    auto_run_on_protocol_complete: bool = False


@api_fields('name', 'version', 'namespace')
@dataclass(frozen=True)
class PluginStatus:
    """A loaded plugin, for health reports."""

    name: str
    version: str
    namespace: str


@api_fields('name', 'version', 'reason')
@dataclass(frozen=True)
class PluginNotLoaded:
    """A plugin that did not load, and why.

    It has no namespace: a plugin is filed under one only by registering,
    which a plugin that never loaded did not do.
    """

    name: str
    version: str
    reason: str


@api_fields('plugin_name', 'namespace', 'hook', 'exc_type', 'message')
@dataclass(frozen=True)
class PluginRuntimeError:
    """A runtime error caught from a plugin handler.

    Distinct from a load failure: the plugin loaded fine, but failed
    while servicing a callback (e.g. auto_run_on_protocol_complete). Reported by
    PluginRegistry.record_runtime_error and kept in
    NamespaceHealth.last_runtime_errors so diagnostic probes can
    attribute fault to the right plugin.
    """

    plugin_name: str
    namespace: str
    hook: str
    exc_type: str
    message: str


@api_fields('namespace', 'loaded', 'last_runtime_errors')
@dataclass(frozen=True)
class NamespaceHealth:
    """Per-namespace snapshot for tech-support + diagnostic probes."""

    namespace: str
    loaded: tuple[PluginStatus, ...]
    last_runtime_errors: tuple[PluginRuntimeError, ...]


@api_fields('namespaces', 'not_loaded')
@dataclass(frozen=True)
class PluginHealth:
    """Every namespace's health and the plugins that did not load, for tech-support reports."""

    namespaces: tuple[NamespaceHealth, ...]
    not_loaded: tuple[PluginNotLoaded, ...]


@api_fields('success', 'outputs', 'message')
@dataclass(frozen=True)
class ProcessorResult:
    """What a post-processing processor answers to the host that called it.

    The host logs a success with its message and what it made, and reports
    a failure to the person with its message.

    Attributes:
        success: Whether the processor completed.
        outputs: The absolute paths of the files it wrote.
        message: A one-line summary for the person using the microscope.
    """

    success: bool
    outputs: tuple[FilePath, ...] = ()
    message: str = ''


# ---------------------------------------------------------------------------
# Per-namespace registries
# ---------------------------------------------------------------------------


class _BaseNamespace:
    """Common state for the four namespace registries."""

    NAMESPACE: str = ''

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # The one record of who is registered here: plugin name -> (its spec,
        # what it registered). health() and the namespace's reads all come
        # from it, so a plugin is listed as loaded exactly while its
        # registration is held.
        self._entries: dict[str, tuple[PluginSpec, Any]] = {}
        self._runtime_errors: list[PluginRuntimeError] = []

    def _record_runtime_error(
        self, plugin_name: str, hook: str, exc_type: str, message: str
    ) -> None:
        self._runtime_errors.append(
            PluginRuntimeError(
                plugin_name=plugin_name,
                namespace=self.NAMESPACE,
                hook=hook,
                exc_type=exc_type,
                message=message,
            )
        )

    def health(self) -> NamespaceHealth:
        with self._lock:
            return NamespaceHealth(
                namespace=self.NAMESPACE,
                loaded=tuple(
                    PluginStatus(name=spec.name, version=spec.version, namespace=self.NAMESPACE)
                    for spec, _registered in self._entries.values()
                ),
                last_runtime_errors=tuple(self._runtime_errors),
            )

    def _assert_unique(self, spec: PluginSpec) -> None:
        if spec.name in self._entries:
            raise PluginRegistrationError(
                'name_taken', f"Plugin '{spec.name}' already registered in '{self.NAMESPACE}'"
            )

    def _discard(self, plugin_name: str) -> None:
        """Remove *plugin_name*'s registration, if it has one here.

        The host's, for a plugin that did not load: what its register()
        registered before it raised goes with it.
        """
        with self._lock:
            self._entries.pop(plugin_name, None)


class UIRegistry(_BaseNamespace):
    """UI-extending plugins. Adds widgets at named mount points."""

    NAMESPACE = 'ui'

    @api(in_process=True)
    def register(
        self, spec: PluginSpec, mount_point: UIMountPoint, builder: Callable[[], object]
    ) -> None:
        """Add a widget at a mount point.

        Args:
            spec: The plugin.
            mount_point: Where the widget is attached.
            builder: Takes no arguments and answers the widget. The host
                calls it at attach time, once the window's widget tree
                exists, not at registration.

        Raises:
            PluginRegistrationError: ``'name_taken'``, the namespace already
                holds a plugin of that name.
        """
        with self._lock:
            self._assert_unique(spec)
            self._entries[spec.name] = (spec, (mount_point, builder))

    def mounts(self) -> tuple[tuple[str, UIMountPoint, Callable[[], object]], ...]:
        """Return (plugin_name, mount_point, builder) tuples for the host
        to attach during widget-tree construction. Returned list is a
        snapshot; subsequent registrations don't appear here."""
        with self._lock:
            return tuple(
                (name, mp, builder) for name, (_spec, (mp, builder)) in self._entries.items()
            )


class PostProcessingRegistry(_BaseNamespace):
    """Operate on saved files. Intern's primary surface.

    register(spec, processor):
        processor: callable
            processor(input_dir, manifest, output_dir) -> ProcessorResult
        Invoked from PluginRegistry.run_protocol_complete_processors() at
        the end of every Full Protocol run when
        spec.auto_run_on_protocol_complete=True.
        Ad-hoc invocation: callers fetch via .get(name) and call directly.
    """

    NAMESPACE = 'post_processing'

    @api(in_process=True)
    def register(
        self,
        spec: PluginSpec,
        processor: Callable[[str, dict, str], ProcessorResult],
    ) -> None:
        """Register a processor of saved files.

        Args:
            spec: The plugin.
            processor: Called ``processor(input_dir, manifest, output_dir)``
                with the folder to read, the run's manifest and the folder to
                write into; answers a ``ProcessorResult``.

        Raises:
            PluginRegistrationError: ``'name_taken'``, the namespace already
                holds a plugin of that name.
        """
        with self._lock:
            self._assert_unique(spec)
            # Store (spec, processor) so .handlers() can hand the spec
            # back to consumers that need its flags (e.g. auto-run gate).
            self._entries[spec.name] = (spec, processor)

    @api(in_process=True)
    def get(self, name: str) -> Callable[[str, dict, str], ProcessorResult] | None:
        """The processor a plugin registered, to call it directly.

        Args:
            name: The plugin's name.

        Returns:
            Callable | None: The processor, or None when no plugin of that
                name is registered here.

        Raises:
            Nothing.
        """
        with self._lock:
            entry = self._entries.get(name)
            return entry[1] if entry is not None else None

    @api(in_process=True)
    def names(self) -> tuple[str, ...]:
        """The names of the plugins registered here.

        Raises:
            Nothing.
        """
        with self._lock:
            return tuple(self._entries.keys())

    def handlers(self) -> tuple[tuple[PluginSpec, Callable], ...]:
        """Return (spec, processor) tuples for every registered plugin.

        Snapshot semantics: subsequent registrations won't appear in
        the returned tuple. Consumers iterate and apply per-plugin
        gates (e.g. spec.auto_run_on_protocol_complete).
        """
        with self._lock:
            return tuple(self._entries.values())


class LiveProcessingRegistry(_BaseNamespace):
    """Per-frame listener plugins. Thin proxy to scope.imaging.

    Registry forwards register / unregister to the canonical listener
    registry on ImagingAPI (one source of truth for the fan-out list).
    This class only keeps a name -> (spec, handler) lookup table so
    unregister-by-plugin-name can resolve to the original handler that
    ImagingAPI was given.

    Built with the session's Lumascope, so no plugin can register before
    there is a scope to forward to.

    Plugin authors call:
        ctx.plugins.live_processing.register(spec, handler)
        ctx.plugins.live_processing.unregister(spec.name)

    Handler signature is cb(frame), the camera's ``Frame`` record. It runs
    on the camera SDK thread; see the imaging.add_frame_listener docstring
    for the budget and the read-only frame contract.
    """

    NAMESPACE = 'live_processing'

    def __init__(self, scope: Any) -> None:
        super().__init__()
        # A plain attribute (not state-on-fan-out) because the listener
        # registry of record lives on ImagingAPI; this is just a lookup
        # channel for unregister-by-name.
        self._scope = scope

    @api(in_process=True)
    def register(self, spec: PluginSpec, frame_handler: Callable[[Frame], None]) -> None:
        """Add a listener on every frame the camera delivers.

        Args:
            spec: The plugin.
            frame_handler: Called ``frame_handler(frame)`` with each
                ``Frame``, on the camera's thread
                (``ImagingAPI.add_frame_listener``).

        Raises:
            PluginRegistrationError: ``'name_taken'``, the namespace already
                holds a plugin of that name.
            HardwareCommandRefusedError: ``'not_connected'``, no camera is
                connected; nothing is registered.
        """
        with self._lock:
            self._assert_unique(spec)
            # Forward to the canonical listener list on ImagingAPI before
            # recording the plugin: a listener the camera refuses raises
            # here, and a plugin recorded first would stay listed as loaded
            # with a handler that never runs, and refuse a retry as a
            # duplicate. The name= param surfaces in the log and the
            # auto-remove warning so L1 can identify which plugin misbehaved.
            self._scope.imaging.add_frame_listener(frame_handler, name=spec.name)
            self._entries[spec.name] = (spec, frame_handler)

    @api(in_process=True)
    def unregister(self, name: str) -> None:
        """Remove the listener a plugin registered; nothing when it registered none.

        Args:
            name: The plugin's name.

        Raises:
            Nothing.
        """
        with self._lock:
            entry = self._entries.pop(name, None)
        if entry is None:
            return
        _spec, handler = entry
        self._scope.imaging.remove_frame_listener(handler)

    def _discard(self, plugin_name: str) -> None:
        # The listener is the registration here: it leaves the camera too.
        self.unregister(plugin_name)

    def names(self) -> tuple[str, ...]:
        """Return the names of all currently-registered live_processing plugins."""
        with self._lock:
            return tuple(self._entries.keys())


class RESTRegistry(_BaseNamespace):
    """REST endpoint plugins: the namespace is reserved, and takes no routes in this release.

    A plugin cannot add REST routes yet, so ``register()`` refuses, and a
    shipped plugin registers none.

    **Stub mode**: while a plugin is being developed, its author can opt
    into ``enable_stub_mode()``. In stub mode, register() accepts the
    spec and records it for ``health()`` reporting (so
    ``ctx.plugins.all_health()`` reflects the registration) but serves
    nothing, so the rest of the plugin -- its load and unregister hooks
    -- can be tried. Stub mode is opt-in and not the default, so a
    shipped plugin's registration is refused, never quietly unserved.
    """

    NAMESPACE = 'rest'

    def __init__(self) -> None:
        super().__init__()
        self._stub_mode: bool = False

    @api(in_process=True)
    def enable_stub_mode(self) -> None:
        """Opt into accepting REST registrations as stubs.

        After enable_stub_mode(), register(spec, router) records the
        registration (visible via health()) and serves nothing, since a
        plugin cannot add REST routes in this release. A plugin author
        uses this while developing a plugin, to try the rest of its
        lifecycle.

        A shipped plugin does not call this: its registration is refused,
        so a route it expects to serve is never silently missing.

        Raises:
            Nothing.
        """
        with self._lock:
            self._stub_mode = True

    def disable_stub_mode(self) -> None:
        """Revert to the default fail-loud register() behavior.

        Existing stubbed registrations remain visible in health(); only
        future register() calls are affected. Symmetric with
        enable_stub_mode for tests that toggle modes.
        """
        with self._lock:
            self._stub_mode = False

    def register(self, spec: PluginSpec, router: Any) -> None:
        if self._stub_mode:
            # The router is held, not mounted: it serves nothing.
            with self._lock:
                self._assert_unique(spec)
                self._entries[spec.name] = (spec, router)
            return
        raise PluginRegistrationError(
            'rest_namespace_reserved',
            'ctx.plugins.rest is reserved, and a plugin cannot add REST routes '
            'in this release, so a shipped plugin registers none. While a '
            'plugin is being developed, ctx.plugins.rest.enable_stub_mode() '
            'records a registration and serves nothing, so the rest of the '
            'plugin can be tried.',
        )


# ---------------------------------------------------------------------------
# Top-level registry container
# ---------------------------------------------------------------------------


@api_fields('ui', 'post_processing', 'live_processing', 'rest')
class PluginRegistry:
    """Single ctx.plugins entry point exposing the four namespaces.

    Attributes:
        ui: Widgets at the window's mount points.
        post_processing: Processors of saved files.
        live_processing: Listeners on every frame the camera delivers.
        rest: Reserved: a plugin cannot add REST routes in this release.

    Lifecycle:
        - Built by the session when its host asks for plugins
          (``ScopeSession.load_plugins``), which then calls load().
        - Drained by unload(), in reverse order, when the host asks
          (``ScopeSession.unload_plugins``) or the session shuts down.
    """

    ui: UIRegistry
    post_processing: PostProcessingRegistry
    live_processing: LiveProcessingRegistry
    rest: RESTRegistry

    def __init__(self, scope: Any) -> None:
        self.ui = UIRegistry()
        self.post_processing = PostProcessingRegistry()
        self.live_processing = LiveProcessingRegistry(scope)
        self.rest = RESTRegistry()
        self._loaded_plugins: list[tuple[str, Any]] = []  # (name, module)
        self._loaded_lock = threading.Lock()
        # The plugins unload() has unregistered, under the same lock: a run's
        # processors that arrive after it are not run, and each is reported
        # as skipped rather than dropped without a word.
        self._unloaded: set[str] = set()
        self._not_loaded: list[PluginNotLoaded] = []
        # Recorded by this discovery and not yet reported, each with the
        # plugin's own exception: load() reports them together when it ends.
        self._unreported: list[tuple[PluginNotLoaded, BaseException | None]] = []

    def _track(self, name: str, module: Any) -> None:
        with self._loaded_lock:
            self._loaded_plugins.append((name, module))

    def _drain(self) -> list[tuple[str, Any]]:
        with self._loaded_lock:
            out = list(self._loaded_plugins)
            self._loaded_plugins.clear()
            self._unloaded.update(name for name, _module in out)
        return out

    def attribute_exception(self, exc_tb) -> str | None:
        """Name the loaded plugin whose code appears in a traceback, or None.

        Walks the traceback frames and matches each frame's file against
        the package directory of every loaded plugin module. Used by the
        app-level crash guard to contain plugin exceptions (popup + log,
        app continues) instead of letting them take down the host --
        plugins are separately versioned and may not be ours to fix, so
        the host cannot assume their handlers are safe.

        Args:
            exc_tb: A traceback object (``sys.exc_info()[2]``).

        Returns:
            str | None: The plugin name when any traceback frame lives
                under a loaded plugin's package directory; None when the
                exception is not attributable to plugin code.
        """
        import os
        import traceback

        with self._loaded_lock:
            roots = []
            for name, module in self._loaded_plugins:
                module_file = getattr(module, '__file__', None)
                if module_file:
                    roots.append((name, os.path.dirname(os.path.abspath(module_file))))
        if not roots:
            return None
        try:
            frames = traceback.extract_tb(exc_tb)
        except Exception:
            return None
        for frame in frames:
            frame_path = os.path.abspath(frame.filename)
            for name, root in roots:
                if frame_path.startswith(root + os.sep):
                    return name
        return None

    def record_load_failure(
        self, name: str, version: str, reason: str, cause: BaseException | None = None
    ) -> None:
        """Record that a plugin did not load, for the discovery's one report.

        The one place a load failure is recorded, so no path that drops a
        plugin can keep it without telling anyone: ``load()`` reports every
        failure recorded here as one outcome when discovery ends. The record
        is kept in ``not_loaded()``.

        Args:
            name: The plugin's name, or its entry point's when it has no spec.
            version: The plugin's version, or '' when it is not known.
            reason: Why it did not load, in words a person can act on.
            cause: The plugin's own exception, when there is one; its
                traceback is carried into the one log record.
        """
        record = PluginNotLoaded(name, version, reason)
        with self._loaded_lock:
            self._not_loaded.append(record)
            self._unreported.append((record, cause))

    def _report_not_loaded(self) -> None:
        """Report every load failure recorded since the last report, as one outcome.

        One, not one per plugin: the reporter merges notices that share a
        title within its window, so plugins failing together would show as
        the first of them alone. Every plugin's own exception is chained, so
        the one log record carries each traceback.
        """
        with self._loaded_lock:
            unreported, self._unreported = self._unreported, []
        if not unreported:
            return
        outcome = PluginNotLoadedError([record for record, _ in unreported])
        causes = [cause for _, cause in unreported if cause is not None]
        if causes:
            outcome.__cause__ = BaseExceptionGroup('the plugins that did not load', causes)
        notifications.report_outcome(outcome, solicited=False, category='Plugins')

    def record_runtime_error(
        self,
        name: str,
        hook: str,
        exc: BaseException | None = None,
        detail: str = '',
    ) -> None:
        """Record that a loaded plugin failed while the host called it, and report it.

        Plugins do not call this. The host catches a plugin's failure and
        feeds it through here, the one place it becomes an outcome, so every
        such failure is both kept for diagnostics and told to the person.
        The report does not depend on the plugin having registered into a
        namespace; the record does.

        Args:
            name: The plugin that failed.
            hook: What the host was calling it for.
            exc: The exception the plugin raised, or None when it failed
                by what it returned.
            detail: The plugin's own account, when it gave one.
        """
        ns = self._find_namespace(name)
        if ns is not None:
            ns._record_runtime_error(
                name,
                hook,
                type(exc).__name__ if exc is not None else '',
                detail or (str(exc) if exc is not None else ''),
            )
        outcome = PluginFailedError(name, hook, detail)
        outcome.__cause__ = exc
        notifications.report_outcome(outcome, solicited=False, category='Plugins')

    def not_loaded(self) -> tuple[PluginNotLoaded, ...]:
        """The plugins that did not load, in the order they were found, for tech-support reports."""
        with self._loaded_lock:
            return tuple(self._not_loaded)

    def health(self) -> PluginHealth:
        """Every namespace's health and the plugins that did not load, in one snapshot."""
        return PluginHealth(namespaces=self.all_health(), not_loaded=self.not_loaded())

    def all_health(self) -> tuple[NamespaceHealth, ...]:
        """Return per-namespace health snapshots for tech-support reports."""
        return tuple(ns.health() for ns in self._namespaces())

    def load(self, host: Any, host_version: str) -> None:
        """Discover the installed plugins, then the built-ins, and register each.

        Entry points (group 'lvp.plugins') go first and the built-ins after
        them, so an installed package claiming a built-in's name keeps it and
        the built-in is the one reported as not loaded. Each plugin's
        register(host) is wrapped; a plugin that does not load, for any
        reason, is recorded and the rest still load; every one recorded is
        reported together, once, when discovery ends. With the
        host's ``no_engineering``, the engineering plugin is left out
        before it is imported: not checked, not registered, not reported.

        Args:
            host: The session hosting the plugins, handed to each plugin as
                its ctx.
            host_version: This LumaViewPro's version, checked against each
                plugin's requires_lvp_version.
        """
        from modules.plugins.builtin import BUILTIN_PLUGINS

        discovered = list(importlib.metadata.entry_points(group=ENTRY_POINT_GROUP))
        count = 0
        # Reported in a finally: a discovery that stops partway still tells
        # the person what it had found would not load.
        try:
            for ep in discovered:
                ep_name = getattr(ep, 'name', '<unknown>')
                if ep_name == ENGINEERING_PLUGIN and host.no_engineering:
                    logger.info(f'[Plugins ] {ep_name} left out: the host has no_engineering')
                    continue
                try:
                    module = ep.load()
                except Exception as e:
                    self.record_load_failure(
                        ep_name, '', f'it could not be imported ({type(e).__name__}: {e})', e
                    )
                    continue
                count += self._load_one(module, ep_name, host, host_version)
            for module in BUILTIN_PLUGINS:
                count += self._load_one(module, module.__name__, host, host_version)
            logger.info(f'[Plugins ] discovery complete -- {count} loaded')
        finally:
            self._report_not_loaded()

    def _load_one(self, module: Any, found_as: str, host: Any, host_version: str) -> int:
        """Register one plugin module with *host*; 1 if it loaded, 0 if not."""
        spec = _extract_spec(module)
        if spec is None:
            self.record_load_failure(
                found_as, '', 'it has no module-level PluginSpec, so it is not a LumaViewPro plugin'
            )
            return 0

        minimum = MINIMUM_PLUGIN_VERSIONS.get(spec.name)
        if minimum is not None:
            have = _parse_semver(spec.version)
            if have is None:
                self.record_load_failure(
                    spec.name,
                    spec.version,
                    f"its version '{spec.version}' cannot be read, and this "
                    f'LumaViewPro needs {minimum} or later',
                )
                return 0
            if have < _parse_semver(minimum):
                self.record_load_failure(
                    spec.name,
                    spec.version,
                    f'version {spec.version} is older than the {minimum} this LumaViewPro needs',
                )
                return 0

        if not is_version_compatible(spec.requires_lvp_version, host_version):
            self.record_load_failure(
                spec.name,
                spec.version,
                f'version {spec.version} needs LumaViewPro {spec.requires_lvp_version}, '
                f'and this is {host_version}',
            )
            return 0

        register_fn = getattr(module, 'register', None)
        if not callable(register_fn):
            self.record_load_failure(spec.name, spec.version, 'it has no register(ctx) function')
            return 0

        held_before = [ns for ns in self._namespaces() if spec.name in ns._entries]
        try:
            register_fn(host)
        except Exception as e:
            self.record_load_failure(
                spec.name, spec.version, f'its register() raised {type(e).__name__}: {e}', e
            )
            # A plugin that did not load has nothing registered: what it
            # registered before the raise is taken back here, whatever its
            # own unregister does. A namespace that held the name before the
            # call holds another plugin's registration, which stays.
            for ns in self._namespaces():
                if ns not in held_before:
                    ns._discard(spec.name)
            # Give the plugin a chance to clean up partial state.
            unregister_fn = getattr(module, 'unregister', None)
            if callable(unregister_fn):
                try:
                    unregister_fn(host)
                except Exception:
                    logger.warning(
                        f'[Plugins ] {spec.name}: unregister after failed register also failed',
                        exc_info=True,
                    )
            return 0

        self._track(spec.name, module)
        logger.info(f'[Plugins ] {spec.name} v{spec.version} loaded')
        return 1

    def unload(self, host: Any) -> None:
        """Call unregister(host) on every loaded plugin, last loaded first.

        The list is emptied first, so a second call unregisters nothing.
        A plugin's unregister failure is logged at WARNING and the rest are
        still unloaded: unloading is part of a close, which must finish.
        """
        for name, module in reversed(self._drain()):
            unregister_fn = getattr(module, 'unregister', None)
            if not callable(unregister_fn):
                continue
            try:
                unregister_fn(host)
                logger.info(f'[Plugins ] {name}: unregister complete')
            except Exception:
                logger.warning(f'[Plugins ] {name}: unregister failed', exc_info=True)

    def run_protocol_complete_processors(
        self,
        input_dir: str,
        manifest: dict,
        output_dir: str,
        files: FilesOutcome,
    ) -> None:
        """Invoke every post_processing plugin that opted in via
        PluginSpec.auto_run_on_protocol_complete=True.

        Called once per Full Protocol run, after all its output files are
        written and its hyperstack build has ended. ``files`` is the run's
        write outcome; on ``INCOMPLETE`` -- some images are not on disk
        -- nothing runs, since a plugin reading the folder as whole would
        build from a partial one. Only a loaded plugin's processor runs: one
        unloaded since is not handed the folder -- the session's close
        unloads plugins first -- and is reported as skipped,
        naming the run (``PluginProcessorSkippedError``). Each plugin's processor runs in
        turn; a processor that raises, returns something other than a
        ProcessorResult, or returns one that reports failure is recorded
        and reported, and does not block the others. A success is logged at
        INFO with what it made: the count of its outputs and their common
        folder, since a plate's stitch makes thousands of files. Never raises.
        """
        if files != FilesOutcome.WRITTEN:
            logger.warning(
                f"[Plugins ] Post-processing auto-run skipped for {input_dir}: the run's "
                f'images were not all written ({files})'
            )
            return
        with self._loaded_lock:
            loaded = {name for name, _module in self._loaded_plugins}
            unloaded = set(self._unloaded)
        for spec, processor in self.post_processing.handlers():
            if not spec.auto_run_on_protocol_complete:
                continue
            if spec.name not in loaded:
                if spec.name in unloaded:
                    notifications.report_outcome(
                        PluginProcessorSkippedError(
                            spec.name, manifest.get('protocol_name') or input_dir, input_dir
                        ),
                        solicited=False,
                        category='Plugins',
                    )
                continue
            try:
                result = processor(input_dir, manifest, output_dir)
            except Exception as e:
                self.record_runtime_error(spec.name, 'auto_run_on_protocol_complete', e)
                continue
            if not isinstance(result, ProcessorResult):
                self.record_runtime_error(
                    spec.name,
                    'auto_run_on_protocol_complete',
                    detail=f'its processor returned {type(result).__name__}, not a ProcessorResult',
                )
                continue
            if result.success:
                logger.info(
                    f'[Plugins ] {spec.name} auto-run succeeded: {result.message} '
                    f'({_what_it_made(result.outputs)})'
                )
            else:
                self.record_runtime_error(
                    spec.name, 'auto_run_on_protocol_complete', detail=result.message
                )

    def _namespaces(self) -> tuple[_BaseNamespace, ...]:
        return (self.ui, self.post_processing, self.live_processing, self.rest)

    def _find_namespace(self, plugin_name: str) -> _BaseNamespace | None:
        for ns in self._namespaces():
            if plugin_name in ns._entries:
                return ns
        return None


def _what_it_made(outputs: tuple[FilePath, ...]) -> str:
    """A processor's outputs in a few words: how many, and the folder they share.

    Read from the paths as text, so outputs on two drives, or a relative
    path among them, cannot make the success log raise.
    """
    if not outputs:
        return 'no files'
    folders = [os.path.dirname(str(output)) + os.sep for output in outputs]
    shared = os.path.commonprefix(folders).rpartition(os.sep)[0]
    return f'{len(outputs)} file(s) in {shared or "more than one folder"}'


# ---------------------------------------------------------------------------
# Version compatibility
# ---------------------------------------------------------------------------


_SEMVER_RE = re.compile(r'^(\d+)\.(\d+)\.(\d+)')
_REQ_RE = re.compile(r'^(>=|>|==|<=|<|~=)?\s*(\d+)\.(\d+)\.(\d+)')


def _parse_semver(s: str) -> tuple[int, int, int] | None:
    """Parse leading semver triple from a string. '4.0.0-beta8' -> (4,0,0)."""
    m = _SEMVER_RE.match(s.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _parse_requirement(req: str) -> tuple[str, tuple[int, int, int]] | None:
    m = _REQ_RE.match(req.strip())
    if not m:
        return None
    op = m.group(1) or '>='
    return op, (int(m.group(2)), int(m.group(3)), int(m.group(4)))


def is_version_compatible(requires: str, host: str) -> bool:
    """Check if the host satisfies the plugin's requires_lvp_version.

    Pre-release suffixes ('-beta8', '-rc1') are stripped before compare.
    A malformed requirement string conservatively returns False so the
    plugin gets visible-rejected rather than silently loaded.
    """
    req = _parse_requirement(requires)
    have = _parse_semver(host)
    if req is None or have is None:
        return False
    op, want = req
    if op == '>=':
        return have >= want
    if op == '>':
        return have > want
    if op == '==':
        return have == want
    if op == '<=':
        return have <= want
    if op == '<':
        return have < want
    if op == '~=':
        # Compatible release: same major, minor >= want.minor
        return have[0] == want[0] and have >= want
    return False


# ---------------------------------------------------------------------------
# Discovery + loading
# ---------------------------------------------------------------------------


def _extract_spec(module: Any) -> PluginSpec | None:
    """Plugins expose a module-level 'spec' attribute."""
    spec = getattr(module, 'spec', None)
    if isinstance(spec, PluginSpec):
        return spec
    return None
