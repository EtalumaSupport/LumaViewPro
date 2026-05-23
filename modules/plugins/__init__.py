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

The host discovers plugins via entry_points group 'lvp.plugins'.
"""

from __future__ import annotations

import copy
import importlib.metadata
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger('lvp_logger')


ENTRY_POINT_GROUP = 'lvp.plugins'

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
    app continues. Host wraps the raise in try/except and fires
    notifications.error so the user sees the failure.
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
    'manual_video.max_fps'). The host fires on_settings_changed only
    when one of those keys changes. Empty tuple = hook never fires.

    auto_run_on_protocol_complete (post_processing namespace only):
    when True, the host invokes the registered processor automatically
    after every protocol run finishes writing files to disk. Defaults
    to False so registration is metadata-only; plugins opt in
    explicitly. See run_protocol_complete_processors().
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


@dataclass(frozen=True)
class PluginStatus:
    """Snapshot of a plugin's load state for health reports."""

    name: str
    version: str
    namespace: str
    loaded: bool
    error: str = ''


@dataclass(frozen=True)
class PluginRuntimeError:
    """A runtime error caught from a plugin handler.

    Distinct from a load failure: the plugin loaded fine, but raised
    while servicing a callback (e.g. on_settings_changed). Logged at
    ERROR and surfaced via NamespaceHealth.last_runtime_errors so
    diagnostic probes can attribute fault to the right plugin.
    """

    plugin_name: str
    namespace: str
    hook: str
    exc_type: str
    message: str


@dataclass(frozen=True)
class NamespaceHealth:
    """Per-namespace snapshot for tech-support + diagnostic probes."""

    namespace: str
    loaded: tuple[PluginStatus, ...]
    failed: tuple[PluginStatus, ...]
    last_runtime_errors: tuple[PluginRuntimeError, ...]


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
        self._failed: list[PluginStatus] = []
        self._runtime_errors: list[PluginRuntimeError] = []
        self._handlers: dict[str, Any] = {}

    def _record_loaded(self, spec: PluginSpec) -> None:
        status = PluginStatus(
            name=spec.name,
            version=spec.version,
            namespace=self.NAMESPACE,
            loaded=True,
        )
        self._loaded[spec.name] = status

    def _record_failed(self, name: str, version: str, error: str) -> None:
        self._failed.append(
            PluginStatus(
                name=name,
                version=version,
                namespace=self.NAMESPACE,
                loaded=False,
                error=error,
            )
        )

    def record_runtime_error(self, plugin_name: str, hook: str, exc: BaseException) -> None:
        """Plugins do not call this directly. The host wraps callbacks
        in try/except and feeds caught exceptions through here so the
        diagnostic surface knows which plugin failed."""
        self._runtime_errors.append(
            PluginRuntimeError(
                plugin_name=plugin_name,
                namespace=self.NAMESPACE,
                hook=hook,
                exc_type=type(exc).__name__,
                message=str(exc),
            )
        )

    def health(self) -> NamespaceHealth:
        with self._lock:
            return NamespaceHealth(
                namespace=self.NAMESPACE,
                loaded=tuple(self._loaded.values()),
                failed=tuple(self._failed),
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
        Invoked from run_protocol_complete_processors() at the end of
        every protocol run when spec.auto_run_on_protocol_complete=True.
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

    def get(self, name: str) -> Optional[Callable]:
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

    Per WAVE7_PHASE_4D5_PLAN sec 9 alignment (2026-05-19): registry
    forwards register / unregister to the canonical listener registry
    on ImagingAPI (Rule 35 -- one source of truth for the fan-out
    list). This class only keeps a name -> (spec, handler) lookup
    table so unregister-by-plugin-name can resolve to the original
    handler that ImagingAPI was given.

    Host wires the live Lumascope via bind_scope() before plugin
    discovery; load_plugins() does this automatically. Register() on
    an unbound registry raises so the failure is loud.

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
        """Wire the live Lumascope. Called by load_plugins() at startup."""
        self._scope = scope

    def register(self, spec: PluginSpec, frame_handler: Callable) -> None:
        if self._scope is None:
            raise PluginRegistrationError(
                'ctx.plugins.live_processing not yet bound to a scope. '
                'The host must call bind_scope(scope) before plugin '
                'discovery; this is normally done inside load_plugins().'
            )
        with self._lock:
            self._assert_unique(spec)
            self._handlers[spec.name] = (spec, frame_handler)
            self._record_loaded(spec)
        # Forward to the canonical listener list on ImagingAPI. The
        # name= param surfaces in WARNING logs and the auto-remove
        # notification body so L1 can identify which plugin misbehaved.
        self._scope.imaging.add_frame_listener(frame_handler, name=spec.name)

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
        - Constructed empty when AppContext is created.
        - Populated by load_plugins(ctx) at app startup after the
          widget tree + AppContext are initialized.
        - Drained by unload_plugins(ctx) at app shutdown, reverse
          order, exceptions swallowed past WARNING.
    """

    def __init__(self) -> None:
        self.ui = UIRegistry()
        self.post_processing = PostProcessingRegistry()
        self.live_processing = LiveProcessingRegistry()
        self.rest = RESTRegistry()
        self._loaded_plugins: list[tuple[str, Any]] = []  # (name, module)
        self._loaded_lock = threading.Lock()

    def _track(self, name: str, module: Any) -> None:
        with self._loaded_lock:
            self._loaded_plugins.append((name, module))

    def _drain(self) -> list[tuple[str, Any]]:
        with self._loaded_lock:
            out = list(self._loaded_plugins)
            self._loaded_plugins.clear()
        return out

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
            ctx: AppContext passed to each plugin's on_settings_changed.
            settings: Full settings dict at the moment of notification
                (post-save snapshot).
            changed_keys: Iterable of dot-path keys that changed in this
                cycle. Empty -> no-op.

        Plugins implement on_settings_changed(ctx, settings) at module
        level (per design doc Sec 4.3). The dispatcher iterates loaded
        plugins regardless of namespace; the per-namespace registries
        retain runtime-error attribution via record_runtime_error.

        Match semantics are prefix: subscribes_to=('manual_video',)
        fires for any changed key under that subtree
        (manual_video.max_fps, manual_video.max_duration, ...).
        subscribes_to=('manual_video.max_fps',) fires only when that
        exact dot-path key changes.

        Exceptions from a plugin handler are logged + recorded but
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
                logger.error(
                    f'[Plugins ] {name}: on_settings_changed raised {type(exc).__name__}: {exc}',
                    exc_info=True,
                )
                ns = self._find_namespace(name)
                if ns is not None:
                    ns.record_runtime_error(name, 'on_settings_changed', exc)

    def _find_namespace(self, plugin_name: str) -> Optional['_BaseNamespace']:
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
    old: Optional[dict],
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


def fire_settings_save_hooks(ctx: Any, new_settings: dict) -> None:
    """Diff new_settings against the ctx-cached baseline + fire plugins.

    Called from MicroscopeSettings.save_settings after the JSON write
    succeeds. The baseline lives at ctx._last_saved_settings_snapshot
    (deepcopied so subsequent in-memory mutations don't poison the
    diff). The first call after startup caches without firing, so
    plugins don't get spurious notifications for the boot-time state.

    The hook is fire-and-forget: any plugin exception is caught inside
    PluginRegistry.notify_settings_changed and never propagates back
    into the save path. If the plugins infrastructure isn't wired
    (e.g. headless test harness without ctx.plugins), the function is
    a no-op.
    """
    if ctx is None or not hasattr(ctx, 'plugins'):
        return
    old = getattr(ctx, '_last_saved_settings_snapshot', None)
    if old is None:
        ctx._last_saved_settings_snapshot = copy.deepcopy(new_settings)
        return
    changed_keys = _diff_settings_keys(old, new_settings)
    if changed_keys:
        ctx.plugins.notify_settings_changed(ctx, new_settings, changed_keys)
    ctx._last_saved_settings_snapshot = copy.deepcopy(new_settings)


# ---------------------------------------------------------------------------
# Version compatibility
# ---------------------------------------------------------------------------


_SEMVER_RE = re.compile(r'^(\d+)\.(\d+)\.(\d+)')
_REQ_RE = re.compile(r'^(>=|>|==|<=|<|~=)?\s*(\d+)\.(\d+)\.(\d+)')


def _parse_semver(s: str) -> Optional[tuple[int, int, int]]:
    """Parse leading semver triple from a string. '4.0.0-beta8' -> (4,0,0)."""
    m = _SEMVER_RE.match(s.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _parse_requirement(req: str) -> Optional[tuple[str, tuple[int, int, int]]]:
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


def _extract_spec(module: Any) -> Optional[PluginSpec]:
    """Plugins expose a module-level 'spec' attribute."""
    spec = getattr(module, 'spec', None)
    if isinstance(spec, PluginSpec):
        return spec
    return None


def _notify_load_failure(ctx: Any, plugin_name: str, reason: str) -> None:
    """Fire a user-facing notification per Rule 14 when a plugin fails to load.

    Best-effort: if notifications aren't wired (e.g. headless test
    harness), the failure is still logged and the function returns.
    """
    try:
        from modules.notification_center import notifications

        notifications.error(
            category='Plugins',
            title='Plugin load failed',
            message=f'{plugin_name} did not load: {reason}. Other features unaffected.',
            source='modules.plugins',
        )
    except Exception:
        logger.exception('[Plugins ] notification_center unavailable')


def load_plugins(ctx: Any) -> None:
    """Discover and load plugins via entry_points group 'lvp.plugins'.

    Called once at app startup after AppContext is initialized and the
    widget tree exists. Each plugin's register(ctx) is wrapped in
    try/except; a failed plugin is logged + notified but does not
    abort the app. The plugin module is tracked so unload_plugins can
    call its unregister(ctx) at shutdown.
    """
    if ctx is None or not hasattr(ctx, 'plugins'):
        logger.error('[Plugins ] load_plugins called without ctx.plugins')
        return

    # Wire the live_processing registry's scope reference before any
    # plugin's register(ctx) can call ctx.plugins.live_processing.register.
    # ctx.scope is expected to be the live Lumascope by this point
    # (LumaViewProApp.build sets it before this call).
    try:
        ctx.plugins.live_processing.bind_scope(getattr(ctx, 'scope', None))
    except Exception:
        logger.exception('[Plugins ] live_processing bind_scope failed')

    host_version = getattr(ctx, 'version', '') or ''
    try:
        discovered = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:
        # Older importlib.metadata returns a dict.
        discovered = importlib.metadata.entry_points().get(ENTRY_POINT_GROUP, [])

    count = 0
    for ep in discovered:
        ep_name = getattr(ep, 'name', '<unknown>')
        try:
            module = ep.load()
        except Exception as e:
            logger.error(
                f'[Plugins ] {ep_name}: import failed: {e}',
                exc_info=True,
            )
            _notify_load_failure(ctx, ep_name, f'import error ({type(e).__name__})')
            continue

        spec = _extract_spec(module)
        if spec is None:
            logger.warning(
                f'[Plugins ] {ep_name}: no module-level PluginSpec, skipping',
            )
            continue

        if not is_version_compatible(spec.requires_lvp_version, host_version):
            logger.warning(
                f'[Plugins ] {spec.name} v{spec.version} requires LVP '
                f'{spec.requires_lvp_version}; have {host_version}; skipping',
            )
            ctx.plugins.ui._record_failed(
                spec.name,
                spec.version,
                f'requires {spec.requires_lvp_version}, have {host_version}',
            )
            continue

        register_fn = getattr(module, 'register', None)
        if not callable(register_fn):
            logger.warning(
                f'[Plugins ] {spec.name}: no register(ctx) function, skipping',
            )
            continue

        try:
            register_fn(ctx)
        except Exception as e:
            logger.error(
                f'[Plugins ] {spec.name}: register() failed: {e}',
                exc_info=True,
            )
            _notify_load_failure(ctx, spec.name, f'{type(e).__name__}: {e}')
            # Give the plugin a chance to clean up partial state.
            unregister_fn = getattr(module, 'unregister', None)
            if callable(unregister_fn):
                try:
                    unregister_fn(ctx)
                except Exception:
                    logger.warning(
                        f'[Plugins ] {spec.name}: unregister after failed register also failed',
                        exc_info=True,
                    )
            continue

        ctx.plugins._track(spec.name, module)
        count += 1
        logger.info(f'[Plugins ] {spec.name} v{spec.version} loaded')

    logger.info(f'[Plugins ] discovery complete -- {count} loaded')


def unload_plugins(ctx: Any) -> None:
    """Call unregister(ctx) on every loaded plugin in reverse order.

    Called from LumaViewProApp.on_stop. Exceptions are caught and
    logged at WARNING; shutdown is not blocked by a plugin's
    unregister failure.
    """
    if ctx is None or not hasattr(ctx, 'plugins'):
        return
    for name, module in reversed(ctx.plugins._drain()):
        unregister_fn = getattr(module, 'unregister', None)
        if not callable(unregister_fn):
            continue
        try:
            unregister_fn(ctx)
            logger.info(f'[Plugins ] {name}: unregister complete')
        except Exception:
            logger.warning(
                f'[Plugins ] {name}: unregister failed',
                exc_info=True,
            )


def run_protocol_complete_processors(
    ctx: Any,
    input_dir: str,
    manifest: dict,
    output_dir: str,
) -> None:
    """Invoke every post_processing plugin that opted in via
    PluginSpec.auto_run_on_protocol_complete=True.

    Called once per protocol run after all output files are written
    to disk. Each plugin's processor runs in turn; per-plugin
    exceptions are caught and logged so one failure does not block
    others or the rest of the completion handler. ProcessorResult is
    logged at INFO on success, WARNING on reported failure.

    Today this is invoked from the UI-side protocol-completion
    handler. When REST-triggered protocol runs land, the dispatcher
    moves down to the orchestration layer so all trigger sources
    benefit uniformly.
    """
    if ctx is None or not hasattr(ctx, 'plugins'):
        return
    for spec, processor in ctx.plugins.post_processing.handlers():
        if not spec.auto_run_on_protocol_complete:
            continue
        try:
            result = processor(input_dir, manifest, output_dir)
        except Exception as e:
            logger.error(
                f'[Plugins ] {spec.name} processor raised {type(e).__name__}: {e}',
                exc_info=True,
            )
            ctx.plugins.post_processing.record_runtime_error(
                spec.name,
                'auto_run_on_protocol_complete',
                e,
            )
            continue
        if not isinstance(result, ProcessorResult):
            logger.warning(
                f'[Plugins ] {spec.name} processor returned '
                f'{type(result).__name__}, expected ProcessorResult',
            )
            continue
        if result.success:
            logger.info(f'[Plugins ] {spec.name} auto-run succeeded: {result.message}')
        else:
            logger.warning(f'[Plugins ] {spec.name} auto-run reported failure: {result.message}')
