# Copyright Etaluma, Inc.
import logging
import os
import pathlib
import subprocess
import sys

from kivy.properties import ListProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.label import Label
from kivy.uix.popup import Popup

from ui.hover_behavior import HoverBehavior
import modules.app_context as _app_ctx

logger = logging.getLogger('LVP.ui.file_dialogs')


# ---------------------------------------------------------------------------
# Kivy folder picker (cross-platform). Replaces the OS-native folder pickers
# (tkinter askdirectory on Windows/Linux, osascript "choose folder" on macOS).
# Both native pickers show only folders, so a folder containing only image
# files renders as an empty mid-pane with "no items match your search" --
# a least-astonishment violation when the user can see files in the folder
# via Explorer/Finder. This picker shows files AND folders, with the user
# confirming the current folder via "Select this folder" or clicking a child
# folder. Clicking a file is interpreted as "use this file's parent folder."
# ---------------------------------------------------------------------------

class FolderPickerPopup(Popup):
    """Kivy folder-picker popup. Shows files and folders; confirms current
    folder, a clicked subfolder, or the parent of a clicked file."""

    def __init__(self, title='Select folder', initial_path=None, on_select=None, **kwargs):
        self._on_select_callback = on_select
        self._result = None

        if initial_path and os.path.isdir(initial_path):
            start_path = str(initial_path)
        else:
            start_path = os.path.expanduser('~')

        layout = BoxLayout(orientation='vertical', padding=8, spacing=6)

        self._path_label = Label(
            text=start_path,
            size_hint_y=None,
            height=24,
            halign='left',
            valign='middle',
            shorten=True,
            shorten_from='left',
        )
        self._path_label.bind(size=lambda w, s: setattr(w, 'text_size', s))
        layout.add_widget(self._path_label)

        self._chooser = FileChooserListView(
            path=start_path,
            dirselect=True,
            show_hidden=False,
        )
        self._chooser.bind(path=self._on_path_change)
        layout.add_widget(self._chooser)

        button_row = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=44,
            spacing=8,
        )
        select_btn = Button(text='Select this folder')
        select_btn.bind(on_release=self._on_confirm)
        button_row.add_widget(select_btn)

        cancel_btn = Button(text='Cancel')
        cancel_btn.bind(on_release=lambda *_: self.dismiss())
        button_row.add_widget(cancel_btn)

        layout.add_widget(button_row)

        super().__init__(
            title=title,
            content=layout,
            size_hint=(0.85, 0.85),
            auto_dismiss=False,
            **kwargs,
        )

    def _on_path_change(self, _instance, path):
        self._path_label.text = str(path)

    def _on_confirm(self, *_):
        # Resolve the target folder:
        #   1. If the user clicked a subfolder, use it.
        #   2. If they clicked a file, use the file's parent folder.
        #   3. Otherwise, use the current browse path (the folder they're in).
        if self._chooser.selection:
            target = self._chooser.selection[0]
            if os.path.isdir(target):
                self._result = target
            else:
                self._result = os.path.dirname(target)
        else:
            self._result = self._chooser.path

        self.dismiss()
        if self._on_select_callback and self._result:
            self._on_select_callback(self._result)


def _open_kivy_folder_picker(initial_dir, on_select, title='Select folder'):
    """Open the Kivy folder picker. Cross-platform; non-blocking; the
    on_select callback fires with the chosen folder path or is not called
    if the user cancels."""
    try:
        popup = FolderPickerPopup(
            title=title,
            initial_path=initial_dir,
            on_select=on_select,
        )
        popup.open()
    except Exception as e:
        logger.error(f'[LVP Main  ] Kivy folder picker error: {e}')


# ---------------------------------------------------------------------------
# macOS native file dialogs via osascript (AppleScript)
# tkinter Tk() crashes on macOS when SDL2 is loaded (cv2 + kivy both ship it).
# plyer requires pyobjus which may not be installed.
# osascript uses native Cocoa panels — no extra dependencies.
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
            ['osascript', '-e', script],
            capture_output=True, text=True, timeout=120
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
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception as e:
        logger.warning(f'[LVP Main  ] macOS folder dialog error: {e}')
    return None


def _platform_native_choose_folder(initial_dir, title='Select folder'):
    """Platform-native folder picker. Returns path string or None.

    Used for the image-save destination folder where the user has
    asked for the OS-native browser (post-processing folder pickers
    still use the in-app Kivy picker so files in the candidate folder
    are visible -- different UX need).
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
            ['osascript', '-e', script],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception as e:
        logger.warning(f'[LVP Main  ] macOS save dialog error: {e}')
    return None


class FileChooseBTN(HoverBehavior, Button):
    context  = StringProperty()
    selection = ListProperty([])

    def choose(self, context):
        logger.info(f'[LVP Main  ] FileChooseBTN.choose({context})')
        self.context = context

        # Show previously selected/default folder
        selected_path = None
        filetypes_tk = None
        if self.context == "load_protocol":
            selected_path = str(pathlib.Path(_app_ctx.ctx.settings['live_folder']))
            filetypes_tk = [('TSV', '.tsv')]
        elif self.context == "load_cell_count_input_image":
            filetypes_tk = [('TIFF', '.tif .tiff')]
        elif self.context == "load_cell_count_method":
            filetypes_tk = [('JSON', '.json')]
        elif self.context == "load_graphing_data":
            filetypes_tk = [('CSV', '.csv')]
        else:
            logger.error(f"Unsupported handling for {self.context}")
            return

        if sys.platform == 'darwin':
            path = _macos_open_file(initial_dir=selected_path, filetypes=filetypes_tk)
            if path:
                self.handle_selection(selection=[path])
            return

        # Windows/Linux: tkinter
        from tkinter import Tk, filedialog
        root = Tk()
        root.attributes('-alpha', 0.0)
        root.attributes('-topmost', True)
        selection = filedialog.askopenfilename(
            parent=root,
            initialdir=selected_path,
            filetypes=filetypes_tk
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
        ctx = _app_ctx.ctx

        if self.selection:
            if self.context == 'load_protocol':
                ctx.motion_settings.ids['protocol_settings_id'].load_protocol(filepath = self.selection[0])

            elif self.context == 'load_cell_count_input_image':
                ctx.cell_count_content.set_preview_source_file(file=self.selection[0])

            elif self.context == 'load_graphing_data':
                ctx.graphing_controls.set_graphing_source(file=self.selection[0])

            elif self.context == 'load_cell_count_method':
                ctx.cell_count_content.load_method_from_file(file=self.selection[0])


class FolderChooseBTN(HoverBehavior, Button):
    context  = StringProperty()
    selection = ListProperty([])

    def choose(self, context):
        logger.info(f'[LVP Main  ] FolderChooseBTN.choose({context})')
        self.context = context

        ctx = _app_ctx.ctx
        settings = ctx.settings

        # Show previously selected/default folder
        if self.context in (
            "apply_stitching_to_folder",
            "apply_composite_gen_to_folder",
            "apply_video_gen_to_folder",
        ):
            selected_path = pathlib.Path(settings['live_folder']) / "ProtocolData"
            if not selected_path.exists():
                selected_path = pathlib.Path(settings['live_folder'])
            selected_path = str(selected_path)
        elif self.context == "apply_zprojection_to_folder":
            # Z-stacks live in TWO canonical places: Manual/Z-Stacks/<ts>/ for
            # the manual ZSTACK button and ProtocolData/<ts>/ for a protocol
            # with Z-stack steps. Opening at live_folder lets the user see
            # both. Fixes #629 (picker was opening one level too deep into
            # ProtocolData, hiding Manual/Z-Stacks behind a navigate-up).
            selected_path = str(pathlib.Path(settings['live_folder']))
        else:
            selected_path = settings['live_folder']

        # live_folder = image-save destination. User wants the OS-native
        # folder browser there (it's a folder-choice, not a folder-inspect
        # action -- the user knows where they want images saved). The
        # post-processing contexts above keep the in-app Kivy picker so
        # the user can see the files inside the folder they're picking,
        # which is the difference that motivated the in-app picker.
        if self.context == 'live_folder':
            chosen = _platform_native_choose_folder(
                initial_dir=selected_path,
                title=f'Select folder ({context})',
            )
            if chosen:
                self.handle_selection(selection=[chosen])
            return

        _open_kivy_folder_picker(
            initial_dir=selected_path,
            on_select=lambda chosen: self.handle_selection(selection=[chosen]),
            title=f'Select folder ({context})',
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
        else:
            return

        if self.context == 'live_folder':
            with ctx.settings_lock:
                settings['live_folder'] = str(pathlib.Path(path).resolve())
        elif self.context == 'apply_cell_count_method_to_folder':
            ctx.cell_count_content.apply_method_to_folder(
                path=path
            )
        elif self.context == 'apply_stitching_to_folder':
            ctx.stitch_controls.run_stitcher(path=pathlib.Path(path))
        elif self.context == 'apply_composite_gen_to_folder':
            ctx.composite_gen_controls.run_composite_gen(path=pathlib.Path(path))
        elif self.context == 'apply_video_gen_to_folder':
            ctx.video_creation_controls.run_video_gen(path=pathlib.Path(path))
        elif self.context == 'apply_zprojection_to_folder':
            ctx.zprojection_controls.run_zprojection(path=pathlib.Path(path))
        else:
            raise Exception(f"on_selection_function(): Unknown selection {self.context}")


class FileSaveBTN(HoverBehavior, Button):
    context  = StringProperty()
    selection = ListProperty([])

    def choose(self, context):
        logger.info('[LVP Main  ] FileSaveBTN.choose()')
        self.context = context
        if self.context == 'saveas_protocol':
            filetypes = [('TSV', '.tsv')]
        elif self.context == 'saveas_cell_count_method':
            filetypes = [('JSON', '.json')]
        elif self.context == 'save_graph':
            filetypes = [('PNG', '.png')]
        else:
            logger.error(f"Unsupported handling for {self.context}")
            return

        selected_path = _app_ctx.ctx.settings['live_folder']

        if sys.platform == 'darwin':
            path = _macos_save_file(initial_dir=selected_path)
            if path:
                self.handle_selection(selection=[path])
            return

        # Windows/Linux: tkinter
        from tkinter import Tk, filedialog
        root = Tk()
        root.attributes('-alpha', 0.0)
        root.attributes('-topmost', True)
        selection = filedialog.asksaveasfilename(
            parent=root,
            initialdir=selected_path,
            filetypes=filetypes
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
        ctx = _app_ctx.ctx

        if self.context == 'saveas_protocol':
            if self.selection:
                ctx.motion_settings.ids['protocol_settings_id'].save_protocol(filepath = self.selection[0])
                logger.info('[LVP Main  ] Saving Protocol to File:' + self.selection[0])

        elif self.context == 'save_graph':
            if self.selection:
                ctx.graphing_controls.save_graph(filepath=self.selection[0])
                logger.info('[LVP Main  ] Saving Graph PNG to File:' + self.selection[0])

        elif self.context == 'saveas_cell_count_method':
            if self.selection:
                logger.info('[LVP Main  ] Saving Cell Count Method to File:' + self.selection[0])
                filename = self.selection[0]
                if os.path.splitext(filename)[1] == "":
                    filename += ".json"
                ctx.cell_count_content.save_method_as(file=filename)
