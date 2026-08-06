# Copyright Etaluma, Inc.
import logging
import os
import pathlib
import subprocess
import sys
import threading
import time

from kivy.clock import Clock
from kivy.properties import ListProperty, StringProperty
from kivy.uix.button import Button

from ui.hover_behavior import HoverBehavior
import modules.app_context as _app_ctx
from modules import gui_logger

logger = logging.getLogger('LVP.ui.file_dialogs')


# Folder-picker contexts that hand work to the file IO executor for
# post-processing. The executor refuses new work while a protocol owns it,
# so these are blocked up front with a message -- otherwise the action's
# progress popup opens but never completes (the task is silently dropped).
# The disabled buttons are the primary stop; this is the backstop for any
# path that reaches the funnel anyway (e.g. an async dialog callback).
_POST_PROCESSING_CONTEXTS = (
    'apply_cell_count_method_to_folder',
    'apply_stitching_to_folder',
    'apply_composite_gen_to_folder',
    'apply_video_gen_to_folder',
    'apply_zprojection_to_folder',
    'apply_quick_enhance_to_folder',
)
# The FILE-choose funnel needs the same backstop. The Quick Enhance preview
# marks its panel busy BEFORE an executor put() that the file executor
# silently drops while a protocol owns it -- the preview callback then never
# fires, busy sticks True, and the panel's disabled binding wedges until
# restart. The other file contexts run synchronously on selection and cannot
# wedge, so only the executor-backed one is listed.
_POST_PROCESSING_FILE_CONTEXTS = ('choose_quick_enhance_target',)


def _zprojection_picker_default_path(live_folder: pathlib.Path) -> str:
    """Return the Z-stack folder the projection picker should open at.

    Z-stacks live in two canonical places:
      - live_folder/Manual/Z-Stacks/<ts>/ -- manual ZSTACK button
        (path defined at ui/zstack.py:234)
      - live_folder/ProtocolData/<ts>/ -- protocol with Z-stack steps

    When exactly ONE of those exists, descend into it (the manual path is
    listed first so a lone manual run is one click away). When BOTH exist,
    open at live_folder instead so neither shadows the other -- always
    descending into Manual/Z-Stacks otherwise hid protocol-produced
    z-stacks even after a protocol run. Neither present also falls back to
    live_folder, so a fresh install never yields an invalid picker target.
    Pure function -- no kivy import; tested via direct invocation in
    tests/test_least_astonishment_fixes.py.
    """
    base = pathlib.Path(live_folder)
    candidates = [
        base / 'Manual' / 'Z-Stacks',
        base / 'ProtocolData',
    ]
    existing = [c for c in candidates if c.exists()]
    if len(existing) == 1:
        return str(existing[0])
    return str(base)


# ---------------------------------------------------------------------------
# Native file dialogs.
# macOS: osascript (AppleScript) Cocoa panels -- tkinter Tk() crashes on
# macOS when SDL2 is loaded (cv2 + kivy both ship it), and plyer requires
# pyobjus which may not be installed.
# Windows/Linux: tkinter, with the Tk root created, used, and destroyed
# entirely inside one worker thread (Tk objects have thread affinity; the
# constraint is per-thread confinement, not "tkinter must be on the main
# thread").
# Every dialog open flows through _run_native_dialog_async -- a blocking
# picker on the Kivy main thread freezes the whole app for as long as the
# panel is open, and the main thread cannot report its own block.
# ---------------------------------------------------------------------------

# Zombie-osascript backstop only, NOT a user-facing limit: a user may browse
# a legitimately-open panel far longer than any short timeout, and a
# timed-out panel is indistinguishable from a cancel. Aligned with the
# dialog-guard expiry below.
_MACOS_DIALOG_TIMEOUT_S = 3600


def _escape_applescript(s):
    """Escape a string for safe interpolation into an AppleScript double-quoted string."""
    return s.replace('\\', '\\\\').replace('"', '\\"')


def _macos_open_file(initial_dir=None, filetypes=None):
    """Show a native macOS open-file dialog. Returns path string or None."""
    script = 'set theFile to choose file'
    clauses = []
    if filetypes:
        # filetypes is list of tuples like [('JSON', '.json')]
        utis = []
        for _, ext in filetypes:
            for e in ext.strip().split():
                utis.append(f'"{e.lstrip(".")}"')
        if utis:
            clauses.append(f'of type {{{", ".join(utis)}}}')
    if initial_dir:
        clauses.append(f'default location POSIX file "{_escape_applescript(initial_dir)}"')
    if clauses:
        script += ' ' + ' '.join(clauses)
    script += '\nPOSIX path of theFile'

    try:
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            timeout=_MACOS_DIALOG_TIMEOUT_S,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception as e:
        logger.warning(f'[LVP Main  ] macOS file dialog error: {e}')
    return None


def _macos_choose_folder(initial_dir=None):
    """Show a native macOS choose-folder dialog. Returns path string or None."""
    script = 'set theFolder to choose folder'
    if initial_dir:
        script += f' default location POSIX file "{_escape_applescript(initial_dir)}"'
    script += '\nPOSIX path of theFolder'

    try:
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            timeout=_MACOS_DIALOG_TIMEOUT_S,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception as e:
        logger.warning(f'[LVP Main  ] macOS folder dialog error: {e}')
    return None


def _macos_choose_file_or_folder(initial_dir=None):
    """Show one native macOS picker that accepts either an image or a folder."""
    script = 'set theItem to choose file or folder'
    if initial_dir:
        script += f' default location POSIX file "{_escape_applescript(initial_dir)}"'
    script += '\nPOSIX path of theItem'

    try:
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            timeout=_MACOS_DIALOG_TIMEOUT_S,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception as e:
        logger.warning(f'[LVP Main  ] macOS file-or-folder dialog error: {e}')
    return None


def _foregrounded_tk_root():
    """Build the invisible Tk root a tkinter picker parents to, foregrounded.

    lift + focus_force after -topmost: parent= and -topmost alone still let
    the picker open buried behind the main window on Windows (the messagebox
    foregrounding precedent, applied here to the picker roots).

    Thread affinity: the caller creates, uses, and destroys this root
    entirely inside one (worker) thread; the root never escapes that
    thread. Destroying it on every path -- including exceptions -- is what
    avoids the Tcl_AsyncDelete process abort from cross-thread deallocation.
    """
    from tkinter import Tk

    root = Tk()
    root.attributes('-alpha', 0.0)
    root.attributes('-topmost', True)
    root.lift()
    root.focus_force()
    return root


def _platform_native_choose_folder(initial_dir, title='Select folder'):
    """Platform-native folder picker (blocking). Returns path string or None.

    Canonical for all FolderChooseBTN contexts as of the #675 broader
    revert. Native pickers show file listings on every modern OS, so
    the prior argument for the in-app Kivy picker ("see files inside
    the candidate folder") no longer justifies the extra UX surface.
    Call only from _run_native_dialog_async's worker.
    """
    if sys.platform == 'darwin':
        return _macos_choose_folder(initial_dir=initial_dir)

    from tkinter import filedialog

    root = _foregrounded_tk_root()
    try:
        path = filedialog.askdirectory(
            parent=root,
            initialdir=initial_dir,
            title=title,
        )
    finally:
        root.destroy()
    return path or None


def _platform_native_open_file(initial_dir, filetypes):
    """Platform-native open-file picker (blocking). Returns path or None.

    Call only from _run_native_dialog_async's worker.
    """
    if sys.platform == 'darwin':
        return _macos_open_file(initial_dir=initial_dir, filetypes=filetypes)

    from tkinter import filedialog

    root = _foregrounded_tk_root()
    try:
        path = filedialog.askopenfilename(
            parent=root,
            initialdir=initial_dir,
            filetypes=filetypes,
        )
    finally:
        root.destroy()
    return path or None


def _platform_native_choose_file_or_folder(initial_dir, filetypes):
    """Return one user-chosen file or folder on every supported platform.

    Cocoa has a single native panel for both target kinds. Tk does not, so
    Windows/Linux present one native yes/no/cancel choice followed by the
    matching native picker. The LVP panel remains a single visible action.
    """
    if sys.platform == 'darwin':
        return _macos_choose_file_or_folder(initial_dir=initial_dir)

    from tkinter import filedialog, messagebox

    root = _foregrounded_tk_root()
    try:
        choose_file = messagebox.askyesnocancel(
            parent=root,
            title='Enhance',
            message='Choose an image file?\nSelect No to choose a folder.',
        )
        if choose_file is True:
            path = filedialog.askopenfilename(
                parent=root,
                initialdir=initial_dir,
                filetypes=filetypes,
            )
        elif choose_file is False:
            path = filedialog.askdirectory(
                parent=root,
                initialdir=initial_dir,
                title='Select folder to enhance',
            )
        else:
            path = ''
    finally:
        root.destroy()
    return path or None


def _platform_native_save_file(initial_dir, filetypes):
    """Platform-native save-file picker (blocking). Returns path or None.

    Call only from _run_native_dialog_async's worker.
    """
    if sys.platform == 'darwin':
        return _macos_save_file(initial_dir=initial_dir)

    from tkinter import filedialog

    root = _foregrounded_tk_root()
    try:
        path = filedialog.asksaveasfilename(
            parent=root,
            initialdir=initial_dir,
            filetypes=filetypes,
        )
    finally:
        root.destroy()
    return path or None


def _macos_save_file(initial_dir=None, default_name=None):
    """Show a native macOS save-file dialog. Returns path string or None."""
    script = 'set theFile to choose file name'
    clauses = []
    if initial_dir:
        clauses.append(f'default location POSIX file "{_escape_applescript(initial_dir)}"')
    if default_name:
        clauses.append(f'default name "{_escape_applescript(default_name)}"')
    if clauses:
        script += ' ' + ' '.join(clauses)
    script += '\nPOSIX path of theFile'

    try:
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            timeout=_MACOS_DIALOG_TIMEOUT_S,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception as e:
        logger.warning(f'[LVP Main  ] macOS save dialog error: {e}')
    return None


# App-wide single-flight record for native dialogs. Module-level, not
# per-button: the programmatic save-protocol path constructs a FRESH
# FileSaveBTN per call, so a per-button flag could never block a stacked
# dialog there. One dialog in flight at a time; other UI stays live.
_dialog_in_flight = {'active': False, 'context': '', 'since': 0.0, 'token': 0}

# Re-click on a busy guard past this many seconds also notifies the user --
# at that age the open panel is probably buried, not being browsed.
_DIALOG_STUCK_NOTIFY_S = 60.0

# A dialog still unresolved after this long is treated as wedged and the
# guard re-arms on its own, so one stuck picker cannot lock out every dialog
# context until app restart. Matches the macOS osascript backstop; the
# tkinter path has no fuse at all, so this expiry is its only unlock. A
# stale dialog that resolves after expiry is dropped by its token.
_DIALOG_GUARD_EXPIRY_S = 3600.0


def _run_native_dialog_async(button, dialog_fn, on_path):
    """Run a blocking native dialog off the Kivy main thread -- the one
    path every dialog open flows through, on every platform.

    Called inline, a native picker blocks the Kivy event loop for the whole
    time the panel is open: the app freezes ("Application Not Responding" /
    beachball) and, because the main thread is the one blocked, nothing can
    even report the freeze. The picker belongs to another process (osascript)
    or a worker-local Tk root, so it does not freeze interaction the way an
    in-app modal would -- which is also why LVP's buttons stay clickable
    behind it; the app-wide single-flight guard stops a second click from
    stacking a second panel.

    dialog_fn runs on a daemon worker thread (the tkinter primitives keep
    their Tk root confined to that thread) and the chosen path is marshalled
    back to the main thread via Clock before on_path runs. on_path is
    invoked only for a non-empty selection. The guard clears ONLY in the
    delivery step, so a second dialog can never open before the first
    dialog's callback has run; a raising primitive still delivers (error
    branch), so it can never leave the guard latched.

    ``button`` supplies the context name for the guard and its logs.
    """
    now = time.monotonic()
    context = getattr(button, 'context', '')
    if _dialog_in_flight['active']:
        elapsed = now - _dialog_in_flight['since']
        if elapsed < _DIALOG_GUARD_EXPIRY_S:
            logger.warning(
                f"[LVP Main  ] Dialog request '{context}' rejected: "
                f"'{_dialog_in_flight['context']}' dialog already in flight "
                f'for {elapsed:.0f}s'
            )
            if elapsed >= _DIALOG_STUCK_NOTIFY_S:
                from modules.notification_center import notifications

                notifications.warning(
                    'File Dialog',
                    'A File Dialog May Already Be Open',
                    'A file dialog appears to be open already -- it may be '
                    'behind the main window. Find and close it, then try '
                    'again.',
                )
            return
        logger.warning(
            f"[LVP Main  ] Dialog guard expired: '{_dialog_in_flight['context']}' "
            f'unresolved for {elapsed:.0f}s; re-arming for {context!r}. If the '
            f'old panel ever resolves, its result will be dropped.'
        )

    _dialog_in_flight['active'] = True
    _dialog_in_flight['context'] = context
    _dialog_in_flight['since'] = now
    _dialog_in_flight['token'] += 1
    token = _dialog_in_flight['token']

    def worker():
        result = None
        error = None
        try:
            result = dialog_fn()
        except Exception as e:
            error = e

        def deliver(_dt):
            if _dialog_in_flight['token'] != token:
                # The guard expired and re-armed while this panel sat
                # unresolved; a newer dialog flow may be live. Dropping the
                # stale result is the only delivery that cannot corrupt it.
                logger.warning(
                    f"[LVP Main  ] Stale dialog result for '{context}' "
                    f'dropped (guard was re-armed while it was open)'
                )
                return
            _dialog_in_flight['active'] = False
            if error is not None:
                logger.error(
                    f"[LVP Main  ] Native dialog '{context}' failed: "
                    f'{type(error).__name__}: {error}'
                )
                from modules.notification_center import notifications

                notifications.error(
                    'File Dialog',
                    'File Dialog Failed',
                    'The file picker could not be opened. Try the button '
                    'again; if it keeps failing, restart LumaViewPro.',
                )
                return
            if result:
                on_path(result)

        Clock.schedule_once(deliver, 0)

    threading.Thread(target=worker, daemon=True).start()


class FileChooseBTN(HoverBehavior, Button):
    context = StringProperty()
    selection = ListProperty([])

    def choose(self, context):
        gui_logger.button('FILE_CHOOSE_OPEN', f'context={context}')
        logger.info(f'[LVP Main  ] FileChooseBTN.choose({context})')
        self.context = context

        # Show previously selected/default folder
        selected_path = None
        filetypes_tk = None
        if self.context == 'load_protocol':
            selected_path = str(pathlib.Path(_app_ctx.ctx.settings['live_folder']))
            filetypes_tk = [('TSV', '.tsv')]
        elif self.context == 'load_cell_count_input_image':
            filetypes_tk = [('TIFF', '.tif .tiff')]
        elif self.context == 'load_quick_enhance_input_image':
            filetypes_tk = [('Images', '.tif .tiff .png .jpg .jpeg .bmp')]
        elif self.context == 'load_cell_count_method':
            filetypes_tk = [('JSON', '.json')]
        elif self.context == 'load_graphing_data':
            filetypes_tk = [('CSV', '.csv')]
        else:
            logger.error(f'Unsupported handling for {self.context}')
            return

        _run_native_dialog_async(
            self,
            lambda: _platform_native_open_file(initial_dir=selected_path, filetypes=filetypes_tk),
            lambda path: self.handle_selection(selection=[path]),
        )

    def handle_selection(self, selection):
        logger.info('[LVP Main  ] FileChooseBTN.handle_selection()')
        if selection:
            self.selection = selection
            self.on_selection_function()

    def on_selection_function(self, *a, **k):
        logger.info('[LVP Main  ] FileChooseBTN.on_selection_function()')
        if self.selection:
            gui_logger.select('FILE_CHOOSE', f'context={self.context} path={self.selection[0]}')
        ctx = _app_ctx.ctx

        if self.context in _POST_PROCESSING_FILE_CONTEXTS and ctx.protocol_running.is_set():
            from modules.notification_center import notifications

            notifications.warning(
                'Post-Processing',
                'Protocol running',
                'Post-processing cannot run while a protocol scan is in '
                'progress. Stop or finish the protocol first, then retry.',
            )
            return

        if self.selection:
            if self.context == 'load_protocol':
                ctx.motion_settings.ids['protocol_settings_id'].load_protocol(
                    filepath=self.selection[0]
                )

            elif self.context == 'load_cell_count_input_image':
                ctx.cell_count_content.set_preview_source_file(file=self.selection[0])

            elif self.context == 'load_quick_enhance_input_image':
                ctx.quick_enhance_controls.set_source_file(file=self.selection[0])

            elif self.context == 'load_graphing_data':
                ctx.graphing_controls.set_graphing_source(file=self.selection[0])

            elif self.context == 'load_cell_count_method':
                ctx.cell_count_content.load_method_from_file(file=self.selection[0])


class FileOrFolderChooseBTN(HoverBehavior, Button):
    """One Enhance entry action that accepts either a supported image or a folder."""

    context = StringProperty()
    selection = ListProperty([])

    def choose(self, context):
        gui_logger.button('FILE_OR_FOLDER_CHOOSE_OPEN', f'context={context}')
        logger.info(f'[LVP Main  ] FileOrFolderChooseBTN.choose({context})')
        self.context = context
        if context != 'choose_quick_enhance_target':
            logger.error('Unsupported file-or-folder context: %s', context)
            return

        initial_dir = str(pathlib.Path(_app_ctx.ctx.settings['live_folder']))
        filetypes = [('Images', '.tif .tiff .png .jpg .jpeg .bmp')]
        _run_native_dialog_async(
            self,
            lambda: _platform_native_choose_file_or_folder(initial_dir, filetypes),
            lambda path: self.handle_selection(selection=[path]),
        )

    def handle_selection(self, selection):
        if selection:
            self.selection = selection
            self.on_selection_function()

    def on_selection_function(self, *a, **k):
        if not self.selection:
            return
        path = pathlib.Path(self.selection[0])
        gui_logger.select('FILE_OR_FOLDER_CHOOSE', f'context={self.context} path={path}')
        ctx = _app_ctx.ctx
        if ctx.protocol_running.is_set():
            from modules.notification_center import notifications

            notifications.warning(
                'Post-Processing',
                'Protocol running',
                'Post-processing cannot run while a protocol scan is in progress. '
                'Stop or finish the protocol first, then retry.',
            )
            return
        if path.is_dir():
            ctx.quick_enhance_controls.set_source_folder(path)
        elif path.is_file():
            ctx.quick_enhance_controls.set_source_file(path)
        else:
            from modules.notification_center import notifications

            notifications.warning('Enhance', 'Selection unavailable', f'Could not open: {path}')


class FolderChooseBTN(HoverBehavior, Button):
    context = StringProperty()
    selection = ListProperty([])

    def choose(self, context):
        gui_logger.button('FOLDER_CHOOSE_OPEN', f'context={context}')
        logger.info(f'[LVP Main  ] FolderChooseBTN.choose({context})')
        self.context = context

        ctx = _app_ctx.ctx
        settings = ctx.settings

        # Show previously selected/default folder
        if self.context in (
            'apply_stitching_to_folder',
            'apply_composite_gen_to_folder',
            'apply_video_gen_to_folder',
        ):
            selected_path = pathlib.Path(settings['live_folder']) / 'ProtocolData'
            if not selected_path.exists():
                selected_path = pathlib.Path(settings['live_folder'])
            selected_path = str(selected_path)
        elif self.context == 'apply_zprojection_to_folder':
            # Z-stacks live in TWO canonical places: Manual/Z-Stacks/<ts>/
            # for the manual ZSTACK button (path defined at
            # ui/zstack.py:234) and ProtocolData/<ts>/ for a protocol
            # with Z-stack steps. Pick the most-specific existing target
            # so a single click reaches the timestamped run, with
            # graceful fallback when the deeper folder was never made.
            selected_path = _zprojection_picker_default_path(
                pathlib.Path(settings['live_folder']),
            )
        else:
            selected_path = settings['live_folder']

        # All FolderChooseBTN contexts use the OS-native folder picker.
        # The earlier in-app Kivy picker was added for post-processing
        # contexts so the user could see the files inside the candidate
        # folder before picking, but native pickers on all supported
        # platforms (macOS Finder, Windows Explorer, Linux GTK) already
        # show file listings -- the Kivy picker was duplicating UX that
        # the OS provides better. Reverted per #675.
        _run_native_dialog_async(
            self,
            lambda: _platform_native_choose_folder(
                initial_dir=selected_path,
                title=f'Select folder ({context})',
            ),
            lambda chosen: self.handle_selection(selection=[chosen]),
        )

    def handle_selection(self, selection):
        logger.info('[LVP Main  ] FolderChooseBTN.handle_selection()')
        if selection:
            self.selection = selection
            self.on_selection_function()

    def on_selection_function(self, *a, **k):
        ctx = _app_ctx.ctx
        settings = ctx.settings
        logger.info('[LVP Main  ] FolderChooseBTN.on_selection_function()')
        if self.selection:
            path = self.selection[0]
            gui_logger.select('FOLDER_CHOOSE', f'context={self.context} path={path}')
        else:
            return

        if self.context in _POST_PROCESSING_CONTEXTS and ctx.protocol_running.is_set():
            from modules.notification_center import notifications

            notifications.warning(
                'Post-Processing',
                'Protocol running',
                'Post-processing cannot run while a protocol scan is in '
                'progress. Stop or finish the protocol first, then retry.',
            )
            return

        if self.context == 'live_folder':
            with ctx.settings_lock:
                settings['live_folder'] = str(pathlib.Path(path).resolve())
        elif self.context == 'apply_cell_count_method_to_folder':
            ctx.cell_count_content.apply_method_to_folder(path=path)
        elif self.context == 'apply_stitching_to_folder':
            ctx.stitch_controls.run_stitcher(path=pathlib.Path(path))
        elif self.context == 'apply_composite_gen_to_folder':
            ctx.composite_gen_controls.run_composite_gen(path=pathlib.Path(path))
        elif self.context == 'apply_video_gen_to_folder':
            ctx.video_creation_controls.run_video_gen(path=pathlib.Path(path))
        elif self.context == 'apply_zprojection_to_folder':
            ctx.zprojection_controls.run_zprojection(path=pathlib.Path(path))
        elif self.context == 'apply_quick_enhance_to_folder':
            ctx.quick_enhance_controls.set_source_folder(path=pathlib.Path(path))
        else:
            raise Exception(f'on_selection_function(): Unknown selection {self.context}')


class FileSaveBTN(HoverBehavior, Button):
    context = StringProperty()
    selection = ListProperty([])

    def choose(self, context):
        gui_logger.button('FILE_SAVE_OPEN', f'context={context}')
        logger.info('[LVP Main  ] FileSaveBTN.choose()')
        self.context = context
        if self.context == 'saveas_protocol':
            filetypes = [('TSV', '.tsv')]
        elif self.context == 'saveas_cell_count_method':
            filetypes = [('JSON', '.json')]
        elif self.context == 'save_graph':
            filetypes = [('PNG', '.png')]
        else:
            logger.error(f'Unsupported handling for {self.context}')
            return

        selected_path = _app_ctx.ctx.settings['live_folder']

        _run_native_dialog_async(
            self,
            lambda: _platform_native_save_file(initial_dir=selected_path, filetypes=filetypes),
            lambda path: self.handle_selection(selection=[path]),
        )

    def handle_selection(self, selection):
        logger.info('[LVP Main  ] FileSaveBTN.handle_selection()')
        if selection:
            self.selection = selection
            self.on_selection_function()

    def on_selection_function(self, *a, **k):
        logger.info('[LVP Main  ] FileSaveBTN.on_selection_function()')
        if self.selection:
            gui_logger.select('FILE_SAVE', f'context={self.context} path={self.selection[0]}')
        ctx = _app_ctx.ctx

        if self.context == 'saveas_protocol':
            if self.selection:
                ctx.motion_settings.ids['protocol_settings_id'].save_protocol(
                    filepath=self.selection[0]
                )
                logger.info('[LVP Main  ] Saving Protocol to File:' + self.selection[0])

        elif self.context == 'save_graph':
            if self.selection:
                ctx.graphing_controls.save_graph(filepath=self.selection[0])
                logger.info('[LVP Main  ] Saving Graph PNG to File:' + self.selection[0])

        elif self.context == 'saveas_cell_count_method' and self.selection:
            logger.info('[LVP Main  ] Saving Cell Count Method to File:' + self.selection[0])
            filename = self.selection[0]
            if os.path.splitext(filename)[1] == '':
                filename += '.json'
            ctx.cell_count_content.save_method_as(file=filename)
