# Copyright Etaluma, Inc.
import logging
import os
import pathlib
import subprocess
import sys
import threading

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
# macOS native file dialogs via osascript (AppleScript)
# tkinter Tk() crashes on macOS when SDL2 is loaded (cv2 + kivy both ship it).
# plyer requires pyobjus which may not be installed.
# osascript uses native Cocoa panels -- no extra dependencies.
# ---------------------------------------------------------------------------


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
            ['osascript', '-e', script], capture_output=True, text=True, timeout=120
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
            ['osascript', '-e', script], capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception as e:
        logger.warning(f'[LVP Main  ] macOS folder dialog error: {e}')
    return None


def _platform_native_choose_folder(initial_dir, title='Select folder'):
    """Platform-native folder picker. Returns path string or None.

    Canonical for all FolderChooseBTN contexts as of the #675 broader
    revert. Native pickers show file listings on every modern OS, so
    the prior argument for the in-app Kivy picker ("see files inside
    the candidate folder") no longer justifies the extra UX surface.
    """
    if sys.platform == 'darwin':
        return _macos_choose_folder(initial_dir=initial_dir)

    from tkinter import Tk, filedialog

    root = Tk()
    root.attributes('-alpha', 0.0)
    root.attributes('-topmost', True)
    path = filedialog.askdirectory(
        parent=root,
        initialdir=initial_dir,
        title=title,
    )
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
            ['osascript', '-e', script], capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception as e:
        logger.warning(f'[LVP Main  ] macOS save dialog error: {e}')
    return None


def _run_macos_dialog_async(button, dialog_fn, on_path):
    """Run a blocking osascript dialog off the Kivy main thread.

    osascript opens the native panel via subprocess.run; called inline it
    blocks the Kivy event loop for the whole time the panel is open, so the
    app beachballs ("Application Not Responding") until the user dismisses
    it. The panel belongs to the osascript process, not LVP, so it does not
    freeze interaction the way an in-app modal would -- which is also why the
    Kivy button stays clickable behind it; the in-flight guard below stops a
    second click from stacking a second panel.

    subprocess.run is thread-safe, so the dialog runs on a daemon thread and
    the chosen path is marshalled back to the main thread (Kivy is
    single-threaded for UI) via Clock before on_path runs. on_path is invoked
    only for a non-empty selection (user picked something, did not cancel).
    """
    if getattr(button, '_dialog_in_flight', False):
        return
    button._dialog_in_flight = True

    def worker():
        result = dialog_fn()

        def deliver(_dt):
            button._dialog_in_flight = False
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

        if sys.platform == 'darwin':
            _run_macos_dialog_async(
                self,
                lambda: _macos_open_file(initial_dir=selected_path, filetypes=filetypes_tk),
                lambda path: self.handle_selection(selection=[path]),
            )
            return

        # Windows/Linux: tkinter
        from tkinter import Tk, filedialog

        root = Tk()
        root.attributes('-alpha', 0.0)
        root.attributes('-topmost', True)
        selection = filedialog.askopenfilename(
            parent=root, initialdir=selected_path, filetypes=filetypes_tk
        )
        root.destroy()

        if selection == '':
            return
        self.handle_selection(selection=[selection])

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
        if sys.platform == 'darwin':
            # Background the osascript panel so the app does not beachball
            # while it is open (see _run_macos_dialog_async). tkinter (below)
            # stays on the main thread -- it is not thread-safe.
            _run_macos_dialog_async(
                self,
                lambda: _macos_choose_folder(initial_dir=selected_path),
                lambda chosen: self.handle_selection(selection=[chosen]),
            )
            return

        chosen = _platform_native_choose_folder(
            initial_dir=selected_path,
            title=f'Select folder ({context})',
        )
        if chosen:
            self.handle_selection(selection=[chosen])

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

        if sys.platform == 'darwin':
            _run_macos_dialog_async(
                self,
                lambda: _macos_save_file(initial_dir=selected_path),
                lambda path: self.handle_selection(selection=[path]),
            )
            return

        # Windows/Linux: tkinter
        from tkinter import Tk, filedialog

        root = Tk()
        root.attributes('-alpha', 0.0)
        root.attributes('-topmost', True)
        selection = filedialog.asksaveasfilename(
            parent=root, initialdir=selected_path, filetypes=filetypes
        )
        root.destroy()

        if selection == '':
            return
        self.handle_selection(selection=[selection])

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
