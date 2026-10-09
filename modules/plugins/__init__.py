# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Plugin platform for LumaViewPro.

Single platform, namespace-scoped registries, lifecycle-aware.
Spec: docs/PLUGIN_API_DESIGN_2026-05-09.md

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
    def on_settings_changed(ctx, settings): ... # optional, fires per spec.subscribes_to

``ctx`` is the ScopeSession hosting the plugin. A host asks its session
for plugins (``session.load_plugins()``); the session discovers them via
entry_points group 'lvp.plugins'.
"""

from __future__ import annotations

import copy
import importlib.metadata
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable, Iterable

from modules.exceptions import (
    PluginFailedError,
    PluginNotLoadedError,
    PluginProcessorSkippedError,
)
from modules.notification_center import notifications
from modules.api_surface import api_fields

logger = logging.getLogger('lvp_logger')


ENTRY_POINT_GROUP = 'lvp.plugins'

# The oldest version of each known plugin this LumaViewPro works with.
# A plugin's requires_lvp_version only guards the other direction, so a
# plugin older than a host change it reaches would otherwise load and
# fail mid-use. A host change the plugin must follow raises the value
# here, in the same commit, to the plugin version that follows it.
MINIMUM_PLUGIN_VERSIONS = {
    'etaluma_engineering': '1.0.48',
}

# What this LumaViewPro does that a plugin may rely on, within its major
# version: no version can say it, since version.txt names the promoted
# release and many trunk commits share one beta number. A commit that
# changes behaviour a plugin relies on raises it by one, in that commit,
# with its row in docs/LumascopeSkills.md ("Plugin API level"). A plugin
# reads it at its own entry through session.plugin_api_level and refuses
# below what it needs; a host older than the level has no such member,
# which the plugin reads as 0.
PLUGIN_API_LEVEL = 5

# Mount points are locked to the set the host knows how to attach.
# Additional names are added when a real consumer needs them, paired
# with a widget-shape contract for that specific mount. Plugins that
# pass an unknown name get a PluginRegistrationError, not a silent
# attach to nothing.
UI_MOUNT_POINTS = frozenset(
    {
        'left_sidebar.accordion',
    }
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PluginRegistrationError(Exception):
    """Raised when a plugin cannot be registered.

    Causes: name collision within a namespace, unknown mount point,
    version mismatch, malformed spec. The plugin is NOT loaded and the
    app continues. The loader catches the raise from the plugin's
    register(ctx) and reports it as that plugin's load failure.
    """


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PluginSpec:
    """Declarative metadata a plugin presents at registration time.

    capabilities lists the API surfaces the plugin uses (dotted paths,
    e.g. 'scope.imaging', 'modules.image_save'). Not enforced as a
    sandbox in 4.x; used by the tech-support report and diagnostic
    probes to record which plugins were loaded when data was collected.

    subscribes_to lists settings-tree keys (dot-path notation, e.g.
    'video.max_fps'). The host fires on_settings_changed only
    when one of those keys changes. Empty tuple = hook never fires.

    auto_run_on_protocol_complete (post_processing namespace only):
    when True, the session invokes the registered processor
    automatically after every Full Protocol run, whoever started it,
    once its files are on disk and its hyperstack build has ended.
    Defaults to False so registration is metadata-only; plugins opt in
    explicitly. See PluginRegistry.run_protocol_complete_processors().
    """

    name: str
    version: str
    requires_lvp_version: str
    description: str
    capabilities: tuple[str, ...] = ()
    subscribes_to: tuple[str, ...] = ()
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
    while servicing a callback (e.g. on_settings_changed). Reported by
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


# Processor result for post_processing namespace.
# Plugins return this from their processor callable so the host knows
# what artifacts to surface in the run-complete dialog and where to
# log success/failure.
@dataclass(frozen=True)
class ProcessorResult:
    success: bool
    outputs: tuple[str, ...] = ()  # absolute paths to produced files
    message: str = ''  # one-line user-facing summary
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-namespace registries
# ---------------------------------------------------------------------------


class _BaseNamespace:
    """Common state for the four namespace registries."""

    NAMESPACE: str = ''

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded: dict[str, PluginStatus] = {}
        self._runtime_errors: list[PluginRuntimeError] = []
        self._handlers: dict[str, Any] = {}

    def _record_loaded(self, spec: PluginSpec) -> None:
        status = PluginStatus(
            name=spec.name,
            version=spec.version,
            namespace=self.NAMESPACE,
        )
        self._loaded[spec.name] = status

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
                loaded=tuple(self._loaded.values()),
                last_runtime_errors=tuple(self._runtime_errors),
            )

    def _assert_unique(self, spec: PluginSpec) -> None:
        if spec.name in self._loaded:
            raise PluginRegistrationError(
                f"Plugin '{spec.name}' already registered in '{self.NAMESPACE}'"
            )


class UIRegistry(_BaseNamespace):
    """UI-extending plugins. Adds widgets at named mount points.

    register(spec, mount_point, builder):
        mount_point: a name from UI_MOUNT_POINTS
        builder: callable returning a Kivy widget. Called by the host
                 at attach time, not at registration time, so the host
                 can defer instantiation until the mount-point widget
                 exists in the tree.
    """

    NAMESPACE = 'ui'

    def register(self, spec: PluginSpec, mount_point: str, builder: Callable[[], Any]) -> None:
        if mount_point not in UI_MOUNT_POINTS:
            raise PluginRegistrationError(
                f"Unknown UI mount point '{mount_point}'. Known: {sorted(UI_MOUNT_POINTS)}"
            )
        with self._lock:
            self._assert_unique(spec)
            self._handlers[spec.name] = (mount_point, builder)
            self._record_loaded(spec)

    def mounts(self) -> tuple[tuple[str, str, Callable[[], Any]], ...]:
        """Return (plugin_name, mount_point, builder) tuples for the host
        to attach during widget-tree construction. Returned list is a
        snapshot; subsequent registrations don't appear here."""
        with self._lock:
            return tuple((name, mp, builder) for name, (mp, builder) in self._handlers.items())


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

    def register(
        self,
        spec: PluginSpec,
        processor: Callable[[str, dict, str], ProcessorResult],
    ) -> None:
        with self._lock:
            self._assert_unique(spec)
            # Store (spec, processor) so .handlers() can hand the spec
            # back to consumers that need its flags (e.g. auto-run gate).
            self._handlers[spec.name] = (spec, processor)
            self._record_loaded(spec)

    def get(self, name: str) -> Callable | None:
        with self._lock:
            entry = self._handlers.get(name)
            return entry[1] if entry is not None else None

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._handlers.keys())

    def handlers(self) -> tuple[tuple[PluginSpec, Callable], ...]:
        """Return (spec, processor) tuples for every registered plugin.

        Snapshot semantics: subsequent registrations won't appear in
        the returned tuple. Consumers iterate and apply per-plugin
        gates (e.g. spec.auto_run_on_protocol_complete).
        """
        with self._lock:
            return tuple(self._handlers.values())


class LiveProcessingRegistry(_BaseNamespace):
    """Per-frame listener plugins. Thin proxy to scope.imaging.

    Registry forwards register / unregister to the canonical listener
    registry on ImagingAPI (one source of truth for the fan-out list).
    This class only keeps a name -> (spec, handler) lookup table so
    unregister-by-plugin-name can resolve to the original handler that
    ImagingAPI was given.

    The registry's load() wires the session's Lumascope via bind_scope()
    before plugin discovery. Register() on an unbound registry raises so
    the failure is loud.

    Plugin authors call:
        ctx.plugins.live_processing.register(spec, handler)
        ctx.plugins.live_processing.unregister(spec.name)

    Handler signature is cb(image, timestamp, chunks). It runs on the
    camera SDK thread; see imaging.add_frame_listener docstring +
    docs/LIVE_PROCESSING_TUTORIAL.md for the budget + don't-mutate
    contract.
    """

    NAMESPACE = 'live_processing'

    def __init__(self) -> None:
        super().__init__()
        # Set by bind_scope() before any register() can succeed. Kept
        # as a plain attribute (not state-on-fan-out) because the
        # listener registry of record lives on ImagingAPI; this is
        # just a lookup channel for unregister-by-name.
        self._scope: Any = None

    def bind_scope(self, scope: Any) -> None:
        """Wire the live Lumascope. Called by PluginRegistry.load()."""
        self._scope = scope

    def register(self, spec: PluginSpec, frame_handler: Callable) -> None:
        if self._scope is None:
            raise PluginRegistrationError(
                'ctx.plugins.live_processing not yet bound to a scope. '
                'bind_scope(scope) must be called before plugin '
                'discovery; PluginRegistry.load() does it.'
            )
        with self._lock:
            self._assert_unique(spec)
            # Forward to the canonical listener list on ImagingAPI before
            # recording the plugin: a listener the camera refuses raises
            # here, and a plugin recorded first would stay listed as loaded
            # with a handler that never runs, and refuse a retry as a
            # duplicate. The name= param surfaces in the log and the
            # auto-remove warning so L1 can identify which plugin misbehaved.
            self._scope.imaging.add_frame_listener(frame_handler, name=spec.name)
            self._handlers[spec.name] = (spec, frame_handler)
            self._record_loaded(spec)

    def unregister(self, name: str) -> None:
        """Remove the listener registered by plugin `name`. No-op if not registered."""
        with self._lock:
            entry = self._handlers.pop(name, None)
        if entry is None or self._scope is None:
            return
        _spec, handler = entry
        self._scope.imaging.remove_frame_listener(handler)

    def names(self) -> tuple[str, ...]:
        """Return the names of all currently-registered live_processing plugins."""
        with self._lock:
            return tuple(self._handlers.keys())


class RESTRegistry(_BaseNamespace):
    """REST endpoint plugins. Name locked; body deferred to REST design session.

    Plugin authors will register a sub-router mounted under
    /plugins/<name>/. The dangerous-command middleware applies to
    plugin endpoints same as core endpoints. Until the REST design
    session locks the URL convention, register() raises so plugins
    don't bake in assumptions that will need to be unwound.

    **Stub mode**: plugin authors who want to prototype against the
    future REST shape can opt into stub mode via ``enable_stub_mode()``.
    In stub mode,
    register() accepts the spec and records it for ``health()``
    reporting (so ``ctx.plugins.all_health()`` reflects the
    registration) but does NOT mount a route. The plugin author can
    iterate on their plugin's lifecycle hooks (load / unregister)
    without needing the real REST surface. Stub mode is opt-in and
    intentionally non-default so production runs continue to fail
    loud on REST registration attempts.
    """

    NAMESPACE = 'rest'

    def __init__(self) -> None:
        super().__init__()
        self._stub_mode: bool = False
        # Stored router objects in stub mode -- not mounted; available
        # for tests / health-introspection only.
        self._stubbed_routers: dict[str, Any] = {}

    def enable_stub_mode(self) -> None:
        """Opt into accepting REST registrations as stubs.

        After enable_stub_mode(), register(spec, router) records the
        registration (visible via health()) but does NOT actually
        mount a route -- the URL convention isn't locked yet. Plugin
        authors use this to prototype the rest of their plugin's
        lifecycle without depending on the unbuilt REST mount machinery.

        Production code MUST NOT call this. The default fail-loud
        contract from non-stub register() exists for a reason: plugins
        that register against the real surface before the URL convention
        is locked will need to be unwound when Phase 1 ships.
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
            self._assert_unique(spec)
            self._record_loaded(spec)
            with self._lock:
                self._stubbed_routers[spec.name] = router
            return
        raise PluginRegistrationError(
            'ctx.plugins.rest is reserved but not yet implemented. '
            'REST URL convention is locked at the REST design session '
            '(tracked at docs/TODO.md). Plan to register here when '
            'REST_API_PLAN.md Phase 1 ships. '
            'Prototype against the future shape via '
            'ctx.plugins.rest.enable_stub_mode() (non-production).'
        )


# ---------------------------------------------------------------------------
# Top-level registry container
# ---------------------------------------------------------------------------


class PluginRegistry:
    """Single ctx.plugins entry point exposing the four namespaces.

    Lifecycle:
        - Built by the session when its host asks for plugins
          (``ScopeSession.load_plugins``), which then calls load().
        - Drained by unload(), in reverse order, when the host asks
          (``ScopeSession.unload_plugins``) or the session shuts down.
    """

    def __init__(self) -> None:
        self.ui = UIRegistry()
        self.post_processing = PostProcessingRegistry()
        self.live_processing = LiveProcessingRegistry()
        self.rest = RESTRegistry()
        self._loaded_plugins: list[tuple[str, Any]] = []  # (name, module)
        self._loaded_lock = threading.Lock()
        # The plugins unload() has unregistered, under the same lock: a run's
        # processors that arrive after it are not run, and each is reported
        # as skipped rather than dropped without a word.
        self._unloaded: set[str] = set()
        self._not_loaded: list[PluginNotLoaded] = []
        # The settings as subscribers last heard them: taken at load, so a
        # change made before the first save is told at that save. Written
        # by load() and settings_saved(), on the host's thread.
        self._settings_baseline: dict | None = None

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
        """Record that a plugin did not load, and report it.

        The one place a load failure becomes an outcome, so no path that
        drops a plugin can keep it without telling anyone. The record is
        kept in ``not_loaded()``.

        Args:
            name: The plugin's name, or its entry point's when it has no spec.
            version: The plugin's version, or '' when it is not known.
            reason: Why it did not load, in words a person can act on.
            cause: The plugin's own exception, when there is one; its
                traceback is carried into the one log record.
        """
        with self._loaded_lock:
            self._not_loaded.append(PluginNotLoaded(name, version, reason))
        outcome = PluginNotLoadedError(name, reason)
        outcome.__cause__ = cause
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
        return (
            self.ui.health(),
            self.post_processing.health(),
            self.live_processing.health(),
            self.rest.health(),
        )

    def notify_settings_changed(
        self,
        ctx: Any,
        settings: dict,
        changed_keys: Iterable[str],
    ) -> None:
        """Fire on_settings_changed on every loaded plugin whose
        subscribes_to prefix-matches any of the changed keys.

        Args:
            ctx: The session, passed to each plugin's on_settings_changed.
            settings: Full settings dict at the moment of notification
                (post-save snapshot).
            changed_keys: Iterable of dot-path keys that changed in this
                cycle. Empty -> no-op.

        Plugins implement on_settings_changed(ctx, settings) at module
        level (per design doc Sec 4.3). The dispatcher iterates loaded
        plugins regardless of namespace; the per-namespace registries
        retain runtime-error attribution via record_runtime_error.

        Match semantics are prefix: subscribes_to=('video',)
        fires for any changed key under that subtree
        (video.max_fps, video.max_duration_seconds, ...).
        subscribes_to=('video.max_fps',) fires only when that
        exact dot-path key changes.

        Exceptions from a plugin handler are recorded and reported but
        never propagate -- one plugin's failure does not block others.
        Runs on the calling thread; plugin handlers must be quick
        enough not to stall the settings-save path.
        """
        changed = set(changed_keys)
        if not changed:
            return
        with self._loaded_lock:
            plugins_snapshot = list(self._loaded_plugins)
        for name, module in plugins_snapshot:
            spec = _extract_spec(module)
            if spec is None or not spec.subscribes_to:
                continue
            if not _any_prefix_match(spec.subscribes_to, changed):
                continue
            handler = getattr(module, 'on_settings_changed', None)
            if not callable(handler):
                continue
            try:
                handler(ctx, settings)
            except Exception as exc:
                self.record_runtime_error(name, 'on_settings_changed', exc)

    def load(self, host: Any, host_version: str) -> None:
        """Discover the installed plugins, then the built-ins, and register each.

        Entry points (group 'lvp.plugins') go first and the built-ins after
        them, so an installed package claiming a built-in's name keeps it and
        the built-in is the one reported as not loaded. Each plugin's
        register(host) is wrapped; a plugin that does not load, for any
        reason, is recorded and reported and the rest still load. The
        settings subscribers will be told about are taken here.

        Args:
            host: The session hosting the plugins, handed to each plugin as
                its ctx.
            host_version: This LumaViewPro's version, checked against each
                plugin's requires_lvp_version.
        """
        from modules.plugins.builtin import BUILTIN_PLUGINS

        self.live_processing.bind_scope(host.scope)
        self._settings_baseline = copy.deepcopy(host.get_settings_snapshot())
        discovered = list(importlib.metadata.entry_points(group=ENTRY_POINT_GROUP))
        count = 0
        for ep in discovered:
            ep_name = getattr(ep, 'name', '<unknown>')
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

        try:
            register_fn(host)
        except Exception as e:
            self.record_load_failure(
                spec.name, spec.version, f'its register() raised {type(e).__name__}: {e}', e
            )
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

    def settings_saved(self, host: Any, settings: dict) -> None:
        """Tell the subscribers what changed since they last heard; *settings* was just saved.

        Diffs against the settings taken at load or at the last save, so a
        change made before the first save is told at it. Each subscriber's
        failure is recorded and reported by notify_settings_changed and
        never reaches the save.
        """
        changed_keys = _diff_settings_keys(self._settings_baseline, settings)
        self._settings_baseline = copy.deepcopy(settings)
        if changed_keys:
            self.notify_settings_changed(host, settings, changed_keys)

    def run_protocol_complete_processors(
        self,
        input_dir: str,
        manifest: dict,
        output_dir: str,
        files: str,
    ) -> None:
        """Invoke every post_processing plugin that opted in via
        PluginSpec.auto_run_on_protocol_complete=True.

        Called once per Full Protocol run, after all its output files are
        written and its hyperstack build has ended. ``files`` is the run's
        write outcome; on ``'incomplete'`` -- some images are not on disk
        -- nothing runs, since a plugin reading the folder as whole would
        build from a partial one. Only a loaded plugin's processor runs: one
        unloaded since, or whose register failed after it registered the
        processor, is not handed the folder; one unloaded since -- the
        session's close unloads plugins first -- is reported as skipped,
        naming the run (``PluginProcessorSkippedError``). Each plugin's processor runs in
        turn; a processor that raises, returns something other than a
        ProcessorResult, or returns one that reports failure is recorded
        and reported, and does not block the others. A success is logged at
        INFO. Never raises.
        """
        if files != 'written':
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
                logger.info(f'[Plugins ] {spec.name} auto-run succeeded: {result.message}')
            else:
                self.record_runtime_error(
                    spec.name, 'auto_run_on_protocol_complete', detail=result.message
                )

    def _find_namespace(self, plugin_name: str) -> _BaseNamespace | None:
        for ns in (
            self.ui,
            self.post_processing,
            self.live_processing,
            self.rest,
        ):
            if plugin_name in ns._loaded:
                return ns
        return None


def _any_prefix_match(
    subscribes_to: tuple[str, ...],
    changed_keys: set[str],
) -> bool:
    """True if any subscription key prefix-matches any changed key.

    Prefix means: subscribes_to='a' matches changed 'a' or 'a.b' or
    'a.b.c' (any descendant under the subtree). 'ab' does NOT match
    'abc' -- the prefix is a full dot-path component.
    """
    for prefix in subscribes_to:
        prefix_dot = prefix + '.'
        for key in changed_keys:
            if key == prefix or key.startswith(prefix_dot):
                return True
    return False


def _diff_settings_keys(
    old: dict | None,
    new: dict,
    prefix: str = '',
) -> set[str]:
    """Return dot-path keys that differ between old and new.

    Treats None old as "no prior snapshot" -- caller should usually
    skip dispatch in that case (initial-save). Recursively descends
    into nested dicts so changes deep inside the settings tree surface
    as their full dot-path (e.g. 'BF.gain' for settings['BF']['gain']).

    Type changes (dict -> scalar or vice versa) count as a change
    of the parent key, not the inner leaves -- a plugin subscribed
    to the parent prefix gets notified.
    """
    if old is None:
        return _flatten_dict_keys(new, prefix)
    changed: set[str] = set()
    all_keys = set(old.keys()) | set(new.keys())
    for k in all_keys:
        full = f'{prefix}.{k}' if prefix else k
        if k not in old or k not in new:
            changed.add(full)
            continue
        old_v = old[k]
        new_v = new[k]
        if isinstance(old_v, dict) and isinstance(new_v, dict):
            changed.update(_diff_settings_keys(old_v, new_v, full))
        elif old_v != new_v:
            changed.add(full)
    return changed


def _flatten_dict_keys(d: dict, prefix: str = '') -> set[str]:
    """Return every leaf-path key in a nested dict as dot-paths."""
    out: set[str] = set()
    for k, v in d.items():
        full = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict_keys(v, full))
        else:
            out.add(full)
    return out


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
