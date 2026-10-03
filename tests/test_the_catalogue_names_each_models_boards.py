# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each scopes.json row names the boards its model has, and the simulator builds from them.

The catalogue named no boards, so the simulator read a model's motor axes as
its LED board: axes meant an EL-0940 scope, none an FX2 scope. That holds
for every row today and is wrong for an FX2 scope with a motorized stage.
Now every row names its ``LEDBoard`` and, exactly when it has motor axes,
its ``MotorBoard``. The simulator is the one reader, so it is where the rule
is checked: a row that breaks it, or names a board the simulator cannot
stand in for, refuses the scope naming the file. Production only warns on a
missing ``LEDBoard``, as on the row's other fields, and never gates on it.
"""

import json
import pathlib
import shutil

import pytest

from drivers.fx2driver import FX2LEDController
from drivers.null_motorboard import NullMotionBoard
from drivers.simulated_camera import SimulatedStall
from drivers.simulated_ledboard import SimulatedLEDBoard
from drivers.simulated_motorboard import SimulatedMotorBoard
from modules import layer_record
from modules.exceptions import InstallationFileError
from tests.scope_fakes import build_scope

REPO_DATA = pathlib.Path(__file__).resolve().parents[1] / 'data'
MODELS = layer_record.load_scope_models()


def _folder(tmp_path, model, mutate) -> pathlib.Path:
    shutil.copytree(REPO_DATA, tmp_path / 'data')
    path = tmp_path / 'data' / 'scopes.json'
    contents = json.loads(path.read_text(encoding='utf-8'))
    mutate(contents['Models'][model])
    path.write_text(json.dumps(contents), encoding='utf-8')
    return tmp_path


def _scope(model, **kwargs):
    return build_scope(simulate=True, sim_model=model, warn_pre_release=False, **kwargs)


@pytest.mark.parametrize('model', sorted(MODELS))
def test_every_row_names_an_led_board_and_a_motor_board_exactly_when_it_has_axes(model):
    entry = MODELS[model]
    assert entry['LEDBoard'] in ('EL-0940', 'FX2')
    if layer_record.model_axes(MODELS, model):
        assert entry['MotorBoard'] == 'EL-0940'
    else:
        assert 'MotorBoard' not in entry


@pytest.mark.parametrize('model', sorted(MODELS))
def test_the_simulated_scope_is_built_with_the_boards_its_row_names(model):
    scope = _scope(model, register_atexit=False)
    try:
        if MODELS[model]['LEDBoard'] == 'FX2':
            assert isinstance(scope._led_driver, FX2LEDController)
        else:
            assert isinstance(scope._led_driver, SimulatedLEDBoard)
        if 'MotorBoard' in MODELS[model]:
            assert isinstance(scope._motion_driver, SimulatedMotorBoard)
        else:
            assert isinstance(scope._motion_driver, NullMotionBoard)
    finally:
        scope.disconnect()


def _drop(field):
    return lambda entry: entry.pop(field)


def _set(field, value):
    return lambda entry: entry.update({field: value})


def test_a_row_naming_an_fx2_with_motor_axes_gets_the_fx2_and_a_motor_board(tmp_path):
    # The shape of an FX2 scope with a motorized stage: its LED board is read
    # from the row, not from whether the row has axes.
    root = _folder(tmp_path, 'LS850', _set('LEDBoard', 'FX2'))

    scope = _scope('LS850', source_path=str(root), register_atexit=False)
    try:
        assert isinstance(scope._led_driver, FX2LEDController)
        assert isinstance(scope._motion_driver, SimulatedMotorBoard)
    finally:
        scope.disconnect()


def test_a_row_naming_an_fx2_with_motor_axes_refuses_a_simulated_camera_stall(tmp_path):
    root = _folder(tmp_path, 'LS850', _set('LEDBoard', 'FX2'))

    with pytest.raises(ValueError, match='simulated with an FX2'):
        _scope('LS850', source_path=str(root), sim_camera_stall=SimulatedStall(0.0, 1.0))


@pytest.mark.parametrize(
    'model, mutate, words',
    [
        ('LS850T', _drop('LEDBoard'), 'names LEDBoard None'),
        ('LS620', _set('LEDBoard', 'LS-9000'), "names LEDBoard 'LS-9000'"),
        ('LS850T', _drop('MotorBoard'), 'gives motor axes and names MotorBoard None'),
        ('LS850', _set('MotorBoard', 'MB-9000'), "gives motor axes and names MotorBoard 'MB-9000'"),
        ('LS620', _set('MotorBoard', 'EL-0940'), 'gives no motor axes but names MotorBoard'),
    ],
    ids=[
        'no-led-board',
        'an-led-board-the-simulator-lacks',
        'axes-without-a-motor-board',
        'a-motor-board-the-simulator-lacks',
        'a-motor-board-without-axes',
    ],
)
def test_a_row_that_breaks_the_rule_refuses_the_simulated_scope_naming_the_file(
    tmp_path, model, mutate, words
):
    root = _folder(tmp_path, model, mutate)

    with pytest.raises(InstallationFileError, match=f"model '{model}' that {words}") as refused:
        _scope(model, source_path=str(root))

    assert refused.value.file_path == root / 'data' / 'scopes.json'


def test_production_warns_on_a_row_missing_its_led_board_by_name(tmp_path):
    from lvp_logger import logger as mock_logger

    root = _folder(tmp_path, 'LS620', _drop('LEDBoard'))
    mock_logger.reset_mock()

    models = layer_record.load_scope_models(root / 'data' / 'scopes.json')

    warned = ' | '.join(str(c) for c in mock_logger.warning.call_args_list)
    assert "'LS620' missing 'LEDBoard'" in warned
    assert 'LS620' in models


def test_production_does_not_warn_on_a_manual_row_without_a_motor_board():
    from lvp_logger import logger as mock_logger

    mock_logger.reset_mock()

    layer_record.load_scope_models(REPO_DATA / 'scopes.json')

    warned = ' | '.join(str(c) for c in mock_logger.warning.call_args_list)
    assert 'MotorBoard' not in warned
