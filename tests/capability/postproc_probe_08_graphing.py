"""CAPABILITY: Graph a cell-count results CSV (axes, labels, trendline) and
save the graph to a file.

GUI equivalent: Post-Processing > Object Plotting popup --
GraphingControls.set_graphing_source (load CSV), set_x_axis / set_y_axis
(choose axes), update_trendline (fit a trendline), save_graph (write the
image). Headless route: session.post_processing.graph_options (the form's
choices) and session.post_processing.graph (draw and save), the members the
popup itself asks.
"""

import sys

import harness


def body(session):
    live = session.settings['live_folder']
    results = f'{live}/results.csv'
    with open(results, 'w') as f:
        f.write('file,count,mean_area\na.tif,3,1.5\nb.tif,5,2.5\nc.tif,8,3.0\n')

    options = session.post_processing.graph_options(results, x='count', y='mean_area')
    harness.check(
        'the form offers the number columns as axes',
        (options.x_axes, options.y_axes) == (('count', 'mean_area'),) * 2,
        f'{options}',
    )
    harness.check('the pair offers a Linear trendline', 'Linear' in options.trendlines)

    saved = session.post_processing.graph(
        results,
        f'{live}/graphs/g.png',
        x='count',
        y='mean_area',
        trendline='Linear',
        title='Area by count',
    )
    harness.check('a script can graph a results CSV', saved.path.is_file(), saved.message)

    try:
        session.post_processing.graph(results, f'{live}/g.v2', x='count', y='mean_area')
    except Exception as refused:  # the refusal's reason is the check
        harness.check(
            'a name without .png is refused',
            getattr(refused, 'reason', None) == 'wrong_extension',
            f'{type(refused).__name__}: {refused}',
        )
    else:
        harness.check('a name without .png is refused', False, 'it was written')


if __name__ == '__main__':
    sys.exit(harness.run(body))
