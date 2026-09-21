"""CAPABILITY: Save a cell-count method to a file, and load one back.

GUI equivalent: Object Analysis popup > Save Method As / Load Method, i.e.
ui/post_processing.py:1158 CellCountControls.save_method_as and
ui/post_processing.py:1167 load_method_from_file (reached from
ui/file_dialogs.py:744 and :537).

The probe asks whether ANY callable below ui/ writes or reads a cell-count
method file, or validates its metadata. A census rather than a guess: it
walks every module under modules/ and reports the matches.
"""

import importlib
import inspect
import pkgutil
import sys

import harness

REPO = harness.REPO
NEEDLES = (
    'save_method',
    'load_method',
    'cell_count_method',
    'method_settings',
    '_validate_method_settings_metadata',
)


def census() -> list[str]:
    hits = []
    for mod in pkgutil.walk_packages([str(REPO / 'modules')], prefix='modules.'):
        try:
            m = importlib.import_module(mod.name)
        except Exception as e:  # a module that will not import cannot serve a script
            hits.append(f'{mod.name}: NOT IMPORTABLE ({e})')
            continue
        for name, obj in vars(m).items():
            if any(n in name for n in NEEDLES):
                hits.append(f'{mod.name}.{name}')
            if inspect.isclass(obj) and obj.__module__ == m.__name__:
                for attr in dir(obj):
                    if any(n in attr for n in NEEDLES):
                        hits.append(f'{mod.name}.{name}.{attr}')
    return hits


def main() -> int:
    hits = census()
    print('callables below ui/ matching', NEEDLES, '->', hits or 'NONE')

    # Also: the method dict the panel starts from. Only ui/ has it.
    # pin-justified: the probe's whole claim is that this capability exists
    # ONLY as widget source; reading it is the evidence, not a seam pin.
    src = (REPO / 'ui' / 'post_processing.py').read_text()
    print("'_get_init_settings' defined in ui/post_processing.py:", '_get_init_settings' in src)
    print('json.dump of the method in ui/post_processing.py:', 'json.dump(self._settings' in src)

    harness.void(
        'a script can read or write a cell-count method file',
        bool(hits),
        'no reader/writer below ui/: the default method dict, the schema/version '
        'stamp, the read, the write and the area/perimeter maxima all live in the '
        'widget, so the method DICT is a module-level contract with no module-level '
        'owner',
    )
    return harness.report()


if __name__ == '__main__':
    sys.exit(main())
