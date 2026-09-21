"""CAPABILITY: Graph a cell-count results CSV (axes, labels, trendline) and
save the graph to a file.

GUI equivalent: Post-Processing > Object Plotting popup --
ui/post_processing.py:869 GraphingControls.set_graphing_source (load CSV),
:584 set_x_axis / :618 set_y_axis (choose axes),
:696 update_trendline (fit one of six trendline types),
:866 save_graph (write the image).

The probe asks whether ANY callable below ui/ builds a graph, fits a
trendline, or saves one. A census, not a guess.
"""

import importlib
import inspect
import pkgutil
import sys

import harness

REPO = harness.REPO
NEEDLES = ('trendline', 'graph', 'plot_', 'savefig', 'scatter')


def census() -> list[str]:
    hits = []
    for mod in pkgutil.walk_packages([str(REPO / 'modules')], prefix='modules.'):
        try:
            m = importlib.import_module(mod.name)
        except Exception as e:
            hits.append(f'{mod.name}: NOT IMPORTABLE ({e})')
            continue
        for name, obj in vars(m).items():
            if any(n in name.lower() for n in NEEDLES):
                hits.append(f'{mod.name}.{name}')
            if inspect.isclass(obj) and obj.__module__ == m.__name__:
                for attr in dir(obj):
                    if any(n in attr.lower() for n in NEEDLES):
                        hits.append(f'{mod.name}.{name}.{attr}')
    return hits


def main() -> int:
    hits = census()
    print('graph-shaped callables below ui/ ->', hits or 'NONE')

    # The trendline maths and the figure both live in the widget body.
    # pin-justified: the probe's whole claim is that the graphing maths exist
    # ONLY as widget source; reading it is the evidence, not a seam pin.
    src = (REPO / 'ui' / 'post_processing.py').read_text()
    for marker in ('np.polyfit', 'plt.subplots', 'plt.savefig', 'pd.read_csv'):
        print(f'{marker!r} in ui/post_processing.py:', marker in src)

    # What a script CAN do today: nothing of the capability. Confirm by
    # attempting the import a script would reach for.
    reachable = True
    try:
        from modules.graphing import Graph  # noqa: F401

        print('modules.graphing imported')
    except ImportError as e:
        print('import modules.graphing ->', e)
        reachable = False

    harness.void(
        'a script can graph a results CSV',
        reachable,
        'modules.graphing does not exist: the CSV parse, the figure, six trendline '
        'branches and the save are all inline in GraphingControls. This is the one '
        'output destination with no implementation below the GUI at all',
    )
    return harness.report()


if __name__ == '__main__':
    sys.exit(main())
