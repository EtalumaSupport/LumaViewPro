"""CAPABILITY: Graph a cell-count results CSV (axes, labels, trendline) and
save the graph to a file.

GUI equivalent: Post-Processing > Object Plotting popup --
GraphingControls.set_graphing_source (load CSV), set_x_axis / set_y_axis
(choose axes), update_trendline (fit a trendline), save_graph (write the
image). The CSV read is modules.post_processing.read_cell_count_results and
the fit is modules.graph_analysis.fit_trendline; the figure and its save are
still the widget's.

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

    # The figure lives in the widget body.
    # pin-justified: the probe's whole claim is that the figure and its save
    # exist ONLY as widget source; reading it is the evidence, not a seam pin.
    src = (REPO / 'ui' / 'post_processing.py').read_text()
    for marker in ('plt.subplots', 'plt.savefig'):
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
        'modules.graphing does not exist: the CSV read and the trendline fit are below '
        'the GUI (read_cell_count_results, graph_analysis.fit_trendline), but the '
        'figure and its save are inline in GraphingControls',
    )
    return harness.report()


if __name__ == '__main__':
    sys.exit(main())
