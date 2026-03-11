# Copyright Etaluma, Inc.
import logging
import os
import pathlib
import subprocess
import sys

from kivy.properties import ListProperty, StringProperty
from kivy.uix.button import Button

from ui.hover_behavior import HoverBehavior
import modules.app_context as _app_ctx

logger = logging.getLogger('LVP.ui.file_dialogs')


# ---------------------------------------------------------------------------
# macOS native file dialogs via osascript (AppleScript)
# tkinter Tk() crashes on macOS when SDL2 is loaded (cv2 + kivy both ship it).
# plyer requires pyobjus which may not be installed.
# osascript uses native Cocoa panels — no extra dependencies.
# ---------------------------------------------------------------------------

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
        clauses.append(f'default location POSIX file "{initial_dir}"')
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
        script += f' default location POSIX file "{initial_dir}"'
    script += '\nPOSIX path of theFolder'

    try:
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().rstrip('/')
    except Exception as e:
        logger.warning(f'[LVP Main  ] macOS folder dialog error: {e}')
    return None


def _macos_save_file(initial_dir=None, default_name=None):
    """Show a native macOS save-file dialog. Returns path string or None."""
    script = 'set theFile to choose file name'
    clauses = []
    if initial_dir:
        clauses.append(f'default location POSIX file "{initial_dir}"')
    if default_name:
        clauses.append(f'default name "{default_name}"')
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
        elif self.context in (
            "apply_zprojection_to_folder",
        ):
            if ctx.last_save_folder is not None:
                selected_path = pathlib.Path(ctx.last_save_folder)
                if not selected_path.exists():
                    selected_path = pathlib.Path(settings['live_folder'])
            else:
                selected_path = pathlib.Path(settings['live_folder'])
            selected_path = str(selected_path)
        else:
            selected_path = settings['live_folder']

        if sys.platform == 'darwin':
            path = _macos_choose_folder(initial_dir=selected_path)
            if path:
                self.handle_selection(selection=[path])
            return

        # Windows/Linux: tkinter
        from tkinter import Tk, filedialog
        root = Tk()
        root.attributes('-alpha', 0.0)
        root.attributes('-topmost', True)
        selection = filedialog.askdirectory(
            parent=root,
            initialdir=selected_path
        )
        root.destroy()

        if selection == '':
            return
        self.handle_selection(selection=[selection])


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
