"""The stitch-registration instrument over a corpus of tiled sets: labels first, then the scores.

#838 and #860 are designed from this probe's report (``stitch_instrument``
holds what it computes). Each set is a protocol run folder, read through
the stitch's own folder loader and laid out through ``overlap_stitcher``
itself (nothing is written), so its tiles and stage positions are the ones
a stitch uses.

Two passes, a person between them:

* ``--propose`` labels every seam of every set (two proposers, phase alone
  where NCC disagrees, a fit over the set, agreement across specimens of
  one overlap),
  writes ``stitch_labels.json`` beside each set as ``proposed`` and a crop
  sheet per set into the output folder. A person reads each sheet and
  marks the set's labels ``confirmed`` (changing any seam to ``no_content``
  or ``unknown`` first).
* ``--score`` scores every rule and placement on each set whose labels are
  confirmed, and on the synthetic scenes (their labels are their
  construction). A set not yet confirmed is named and not scored.

With no corpus it scores the synthetic scenes alone, the form every probe
runs in under the simulator.

    python tests/capability/postproc_probe_12_stitch_registration.py --propose CORPUS
    python tests/capability/postproc_probe_12_stitch_registration.py --score CORPUS
    python tests/capability/postproc_probe_12_stitch_registration.py
"""

import argparse
import datetime
import json
import pathlib
import sys

import harness
from harness import check, figure, make_session, report

import stitch_instrument as si

LABELS_FILE = 'stitch_labels.json'


def _arguments():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--propose', action='store_true')
    mode.add_argument('--score', action='store_true', help='the default')
    parser.add_argument(
        'corpus',
        type=pathlib.Path,
        nargs='?',
        default=None,
        help='a folder of tiled sets, one run folder each; none scores the synthetic scenes alone',
    )
    parser.add_argument('--out', type=pathlib.Path, default=None)
    return parser.parse_args()


def _run_folders(corpus):
    return sorted(p.parent for p in corpus.rglob('protocol_record.tsv'))


def _load_set(session, folder):
    """A set's tiles at stage, and today's two floods, through the stitch's own path."""
    import modules.image_utils as image_utils
    from modules.common_utils import PostFunction
    from modules.protocol_post_processing_helper import ProtocolPostProcessingHelper
    from modules.stitching_core import overlap_stitcher

    loaded = ProtocolPostProcessingHelper().load_folder(
        path=folder, tiling_grids=session.scope.protocols.tiling_grids()
    )
    if loaded['status'] is False:
        raise RuntimeError(f'{folder}: {loaded["message"]}')
    root = loaded['root_path']
    df = loaded['images_df']
    df = df[df[PostFunction.STITCHED.value] == False]  # noqa: E712 -- pandas mask
    groups = df['Tile Group ID'].unique()
    if len(groups) != 1:
        raise RuntimeError(f'{folder}: {len(groups)} tile groups; a set is one')
    pixel_size_um = image_utils.agreed_facts(
        image_utils.read_file_record(root / name) for name in df['Filepath']
    ).get('pixel_size_um')
    columns = [
        c
        for c in ('Filepath', 'X', 'Y', 'Objective', 'Color', 'Well', 'Tile Group ID')
        if c in df.columns
    ]
    floods = {}
    tiles = None
    for name, estimator in si.ESTIMATORS.items():
        result = overlap_stitcher(
            path=root,
            df=df[columns],
            pixel_size_um=pixel_size_um,
            output_file_loc=None,
            estimator=estimator,
            channel_colors=session.channel_colors(),
        )
        if not result['status']:
            raise RuntimeError(f'{folder}: {result["error"]}')
        registered = result['metadata']['registered_tiles']
        floods[name] = [
            (float(t['registered_x_px']), float(t['registered_y_px'])) for t in registered
        ]
        tiles = tiles or [
            {'tile': t['tile'], 'x_px': t['x_px'], 'y_px': t['y_px']} for t in registered
        ]
    overlap = float(df['Overlap'].iloc[0]) if 'Overlap' in df.columns else None
    return tiles, floods, {'pixel_size_um': pixel_size_um, 'overlap': overlap}


def _overlap_of(folder):
    # The probe that captured the corpus names each set <specimen>_<grid>_<NN>pct.
    for part in folder.parent.name.split('_'):
        if part.endswith('pct'):
            return part
    return 'unknown overlap'


def _label_to_json(label):
    return {
        'kind': label.kind,
        'dx': round(label.dx, 3),
        'dy': round(label.dy, 3),
        'why': label.why,
        'phase_alone': label.phase_alone,
    }


def _propose(session, folders, out):
    import cv2

    by_overlap = {}
    sets = {}
    for folder in folders:
        tiles, _floods, facts = _load_set(session, folder)
        labels = si.propose_labels(tiles)
        sets[folder] = (tiles, labels, facts)
        by_overlap.setdefault(_overlap_of(folder), {})[str(folder)] = {
            s.key: lab for s, lab in labels.items()
        }
    checked = {}
    for group in by_overlap.values():
        checked.update(si.across_specimens(group))
    for folder, (tiles, labels, facts) in sets.items():
        final = {s: checked[str(folder)][s.key] for s in labels}
        answers = {s: {name: si.answer(name, tiles, s) for name in si.ESTIMATORS} for s in final}
        name = folder.parent.name
        sheet = out / f'{name}_seams.png'
        cv2.imwrite(str(sheet), si.crop_sheet(tiles, answers, final))
        (folder / LABELS_FILE).write_text(
            json.dumps(
                {
                    'status': 'proposed',
                    'confirmed': None,
                    'proposed_at': datetime.datetime.now()
                    .astimezone()
                    .isoformat(timespec='seconds'),
                    'facts': facts,
                    'seams': {s.key: _label_to_json(lab) for s, lab in final.items()},
                },
                indent=1,
            ),
            encoding='utf-8',
        )
        kinds = [lab.kind for lab in final.values()]
        figure(f'{name}.labels', {k: kinds.count(k) for k in ('offset', 'no_content', 'unknown')})
        figure(f'{name}.sheet', str(sheet))


def _score_one(name, tiles, labels, floods, lines):
    seam_list = list(labels)
    truth = si.labelled_positions(tiles, labels)
    lines.append(f'\n### {name}\n')
    lines.append(
        '| rule | right | wrong | missed | declined right | border (Quality) | zero or out (Fast) | flood | joint | fitted stage | stage |'
    )
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    unknown = []
    stage_error, _ = si.worst_tile_error(si.place_stage(tiles), truth)
    for rule in si.RULES:
        answers = {s: si.answer(rule.estimator, tiles, s) for s in seam_list}
        scored = si.score_seams(rule, answers, labels)
        unknown = scored.unknown
        flood, _ = si.worst_tile_error(si.place_flood(rule, tiles), truth)
        joint, _ = si.worst_tile_error(si.place_joint(rule, tiles, answers), truth)
        fitted_positions, model = si.place_fitted_stage(rule, tiles, answers)
        fitted, _ = si.worst_tile_error(fitted_positions, truth)
        c = scored.counts
        lines.append(
            f'| {rule.name} | {c["right"]} | {c["wrong"]} | {c["missed"]} | {c["declined right"]} | '
            f'{scored.border} | {scored.zero_or_out} | {flood:.1f} | {joint:.1f} | {fitted:.1f} | {stage_error:.1f} |'
        )
        figure(
            f'{name}.{rule.name}',
            {
                **c,
                'flood_px': round(flood, 1),
                'joint_px': round(joint, 1),
                'fitted_px': round(fitted, 1),
            },
        )
        if floods is not None and rule.name.startswith('today'):
            shipped = floods[rule.estimator]
            check(
                f'{name}: the harness flood under "{rule.name}" is the shipped stitch',
                si.place_flood(rule, tiles) == shipped,
            )
    model = si.fit_stage_model(
        {s: (lab.dx, lab.dy) for s, lab in labels.items() if lab.kind == 'offset'}
    )
    if model is not None:
        lines.append(
            f'\nStage model from the labels: scale x {model.scale_x:.5f}, y {model.scale_y:.5f}, '
            f'rotation {model.rotation_deg:.3f} deg, {model.seams_fitted} seams.'
        )
    unplaced = sum(t is None for t in truth)
    if unplaced:
        lines.append(
            f'\n{unplaced} tile(s) have no labelled position and are left out of the tile errors.'
        )
    if unknown:
        lines.append('\nNot scored (unknown):\n' + '\n'.join(f'- {u}' for u in unknown))


def _score(session, folders, out):
    lines = [
        '# Stitch registration instrument report',
        '',
        f"Window {si.PRODUCTION_WINDOW_PX} px (production's), right within {si.RIGHT_WITHIN_PX} px on every axis.",
        'Tile errors are the worst tile\'s distance from its labelled position, px. "flood" is production\'s '
        'flood under the rule: a declined tile moves with its parent, it is not stage.',
        '',
        '## Real sets',
    ]
    for folder in folders:
        name = folder.parent.name
        record = folder / LABELS_FILE
        if not record.exists():
            figure(f'{name}.not_scored', 'no labels: run --propose')
            lines.append(f'\n### {name}\n\nNot scored: no labels yet.')
            continue
        stored = json.loads(record.read_text(encoding='utf-8'))
        if stored.get('status') != 'confirmed':
            figure(f'{name}.not_scored', 'labels proposed, not confirmed')
            lines.append(f'\n### {name}\n\nNot scored: its labels are proposed, not confirmed.')
            continue
        tiles, floods, _facts = _load_set(session, folder)
        by_key = {s.key: s for s in si.seams(tiles)}
        labels = {
            by_key[k]: si.Label(v['kind'], v['dx'], v['dy'], v['why'], v['phase_alone'])
            for k, v in stored['seams'].items()
        }
        _score_one(name, tiles, labels, floods, lines)
    lines.append('\n## Synthetic scenes (labels are their construction)')
    for scene in si.SCENES:
        for overlap in (0.1, 0.2):
            tiles, labels = si.synthetic_set(scene, overlap)
            _score_one(f'{scene}, {int(overlap * 100)} %', tiles, labels, None, lines)
    path = out / 'report.md'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    figure('report', str(path))


def main():
    args = _arguments()
    out = args.out.resolve() if args.out else harness.probe_dir('stitch_registration')
    out.mkdir(parents=True, exist_ok=True)
    folders = []
    if args.corpus is not None:
        corpus = args.corpus.resolve()
        folders = _run_folders(corpus)
        check('the corpus holds tiled sets', bool(folders), str(corpus))
    elif args.propose:
        check('--propose is given a corpus to label', False)
    session, _live = make_session('stitch_registration')
    try:
        if args.propose:
            _propose(session, folders, out)
        else:
            _score(session, folders, out)
    finally:
        session.shutdown()
    return report()


if __name__ == '__main__':
    sys.exit(main())
