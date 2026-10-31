# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The EL-0940 serial drivers: the exchange, its lock, the LED acks, the
motor board's moves, reads, parse and scale.

The LED and motor drivers run over the shipped firmware behind the
emulated port (`tests/wire_boards.py`), so every reply is the firmware's
own. A file-wide board serves the tests that leave it as they found it; a
test that pulls the cable, ends the connection or needs another firmware
builds its own.

A claim about the port itself -- a write that times out, a port that
fails, a board that sends nothing at all -- runs the production driver over
a scripted port behind its own ``backend=`` seam (`tests/scripted_port.py`).
Left on scripted replies: an LED firmware that reports a version (the
emulation carries only the field LED firmware, which reports none).
"""

import logging
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import serial

# Heavy deps are mocked by tests/conftest.py at module-import time.

import drivers.serialboard as serialboard
from drivers.exceptions import BoardNoReplyError, BoardPortFailedError, PartAbsentError
from drivers.ledboard import LEDBoard
from drivers.serialboard import SerialBoard
from drivers.sim_wire.mp.tmc5072 import XACTUAL, XTARGET
from tests.log_capture import messages, records_of
from tests.motorconfig_fixtures import SHIPPED_MOTOR_DEFAULTS
from tests.scripted_port import LED_PID, LED_VID, ScriptedBackend, ScriptedLink
from tests.wire_boards import led_on_the_wire, motor_on_the_wire

# Past the LED driver's 0.3 s wait for a reply line.
_LATE_S = 0.6


@pytest.fixture(scope='module')
def _led_board():
    board, sim = led_on_the_wire(oracle=True)
    yield board, sim
    board.disconnect()


@pytest.fixture
def led(_led_board):
    """The file's LED board, every LED turned off after the test."""
    yield _led_board
    _led_board[0].leds_off()


@pytest.fixture(scope='module')
def motor():
    """The file's LS850T motor board on the field firmware, its oracle on."""
    board, sim = motor_on_the_wire(oracle=True)
    yield board, sim
    board.disconnect()


@pytest.fixture(scope='module')
def motor_3_0():
    board, sim = motor_on_the_wire(dialect='3.0')
    yield board, sim
    board.disconnect()


@contextmanager
def _answered_late(board, sim):
    """The body's first exchange gets no answer in time: the board's reply is
    held past the driver's wait and lands in the port's buffer after it,
    before the body's next command is written."""
    port = board.driver
    landed = port.lines_landed()
    sim.delay_next_reply(_LATE_S)
    yield
    assert port.wait_for_lines(landed + 2), 'the held reply never landed'


def _exchange_in_flight(board, sim, command):
    """Start *command*'s exchange on its own thread with its reply held back,
    and return once the command is on the wire, the exchange holding the
    serial lock until the reply lands: (the thread, its answers, the lines
    landed before it)."""
    port = board.driver
    landed = port.lines_landed()
    port.wrote.clear()
    sim.delay_next_reply(0.3)
    answers = []
    thread = threading.Thread(
        target=lambda: answers.append(board.exchange_command(command, timeout=5))
    )
    thread.start()
    assert port.wrote.wait(5), 'the exchange never wrote'
    return thread, answers, landed


def _writes_to(sim, axis, reg):
    return [w.value for w in sim.take_writes() if w.axis == axis and w.reg == reg]


# ---------------------------------------------------------------------------
# The exchange and its lock
# ---------------------------------------------------------------------------


@pytest.mark.firmware_tier
class TestTheLedExchange:
    def test_an_exchange_returns_the_answer_not_the_echo(self, led):
        # The field firmware echoes 'RE: LED0_100' before it answers.
        board, _sim = led
        assert board.exchange_command('LED0_100') == 'LED 0 set to 100 mA.'

    def test_a_write_only_command_waits_for_an_exchange_in_flight(self, led):
        board, sim = led
        port = board.driver
        thread, answers, landed = _exchange_in_flight(board, sim, 'LED0_100')
        board.led_on_fast(channel=1, mA=50)
        # The fast write had the lock only once the exchange had its reply.
        assert port.lines_landed() >= landed + 2
        thread.join(5)
        assert answers == ['LED 0 set to 100 mA.']
        assert port.wait_for_lines(landed + 4), "the fast write's reply never landed"

    def test_concurrent_exchanges_each_get_their_own_answer(self, led):
        board, _sim = led
        answers = {}
        threads = [
            threading.Thread(
                target=lambda ch=ch: answers.__setitem__(ch, board.exchange_command(f'LED{ch}_100'))
            )
            for ch in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        assert answers == {ch: f'LED {ch} set to 100 mA.' for ch in range(5)}


@pytest.mark.firmware_tier
class TestTheMotorExchange:
    def test_an_exchange_returns_the_firmwares_one_line_answer(self, motor):
        board, _sim = motor
        assert board.exchange_command('INFO') == (
            'EL-0940-01 Integrated Mainboard Firmware:     2024-09-10'
        )

    def test_an_exchange_over_a_pulled_cable_raises_and_closes_the_port(self):
        """A lost link is not silence: the exchange raises the port's
        failure, and the port is closed."""
        board, sim = motor_on_the_wire()
        sim.unplug()
        with pytest.raises(BoardPortFailedError, match=r'^INFO: the port failed'):
            board.exchange_command('INFO')
        assert board.driver is None

    def test_a_disconnect_waits_for_the_exchange_in_flight(self):
        board, sim = motor_on_the_wire()
        port = board.driver
        thread, answers, landed = _exchange_in_flight(board, sim, 'INFO')
        board.disconnect()
        # The disconnect had the lock only once the exchange had its reply.
        assert port.lines_landed() >= landed + 1
        thread.join(5)
        assert answers == ['EL-0940-01 Integrated Mainboard Firmware:     2024-09-10']
        assert board.driver is None


# ---------------------------------------------------------------------------
# The LED board's writes and acks
# ---------------------------------------------------------------------------


def _fast(board, call, lines=2):
    """Make a write-only *call* and wait for the board's *lines* of answer to
    land, its echo among them: the DAC's state is reported before each line
    of the answer is."""
    port = board.driver
    landed = port.lines_landed()
    call()
    assert port.wait_for_lines(landed + lines), "the fast write's reply never landed"


@pytest.mark.firmware_tier
class TestTheLedWrites:
    def test_led_on_fast_drives_what_led_on_drives(self, led):
        board, sim = led
        board.led_on(channel=1, mA=50)
        answered = sim.state('DAC')[1]
        board.led_off(channel=1)
        _fast(board, lambda: board.led_on_fast(channel=1, mA=50))
        assert sim.state('DAC')[1] == answered

    def test_led_off_fast_drives_what_led_off_drives(self, led):
        board, sim = led
        board.led_on(channel=4, mA=50)
        board.led_off(channel=4)
        answered = sim.state('DAC')[4]
        board.led_on(channel=4, mA=50)
        _fast(board, lambda: board.led_off_fast(channel=4))
        assert sim.state('DAC')[4] == answered

    def test_leds_off_fast_drives_what_leds_off_drives(self, led):
        board, sim = led
        board.led_on(channel=1, mA=50)
        board.led_on(channel=3, mA=20)
        board.leds_off()
        answered = sim.state('DAC')
        board.led_on(channel=1, mA=50)
        board.led_on(channel=3, mA=20)
        # The field firmware answers LEDS_OFF a line a channel, 0 to 5.
        _fast(board, board.leds_off_fast, lines=7)
        assert sim.state('DAC') == answered

    def test_get_status_is_none_and_asks_the_board_nothing(self, led):
        # The LED firmware has no STATUS command.
        board, _sim = led
        board.driver.wrote.clear()
        assert board.get_status() is None
        assert not board.driver.wrote.is_set()

    def test_wait_until_on_is_false_and_asks_the_board_nothing(self, led):
        board, _sim = led
        board.driver.wrote.clear()
        assert board.wait_until_on() is False
        assert not board.driver.wrote.is_set()

    def test_a_blocking_led_on_polls_past_an_answer_that_came_late(self, led):
        board, sim = led
        board.led_on(channel=3, mA=100)
        lit = sim.state('DAC')[3]
        board.leds_off()
        sim.delay_next_reply(_LATE_S)
        board.led_on(channel=3, mA=100, block=True)
        assert sim.state('DAC')[3] == lit

    @pytest.mark.parametrize(
        'command,call',
        [
            ('LEDS_ENT', lambda b: b.leds_enable()),
            ('LEDS_ENF', lambda b: b.leds_disable()),
            ('LED3_5', lambda b: b.led_on(channel=3, mA=5)),
            ('LED3_OFF', lambda b: b.led_off(channel=3)),
            ('LEDS_OFF', lambda b: b.leds_off()),
        ],
    )
    def test_a_write_answered_past_the_wait_is_not_an_ack(self, led, command, call):
        """A wedged board accepts the write and does not answer in time.

        Treating that as success cleared the error field and left the API
        reporting a channel lit while the hardware was dark -- the 18-minute
        wedge in the SNlogs-2026-05-14-134147 bundle.
        """
        board, sim = led
        with (
            _answered_late(board, sim),
            pytest.raises(BoardNoReplyError, match=rf'^{command}: no reply from the board'),
        ):
            call(board)

    def test_concurrent_led_on_and_led_off_are_each_answered(self, led):
        board, _sim = led
        errors = []

        def toggle(write):
            try:
                for _ in range(50):
                    write()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=toggle, args=(lambda: board.led_on(channel=0, mA=100),)),
            threading.Thread(target=toggle, args=(lambda: board.led_off(channel=0),)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        assert not errors, f'Concurrent writes raised: {errors}'
        assert board.exchange_command('LED2_OFF') == 'LED 2 has been turned off'


@pytest.mark.firmware_tier
class TestSerialDesyncRecovery:
    """A reply that lands after its command gave up sits in the port's buffer;
    read as the next command's answer, every answer after it is one behind."""

    def test_after_each_late_reply_the_next_command_gets_its_own_answer(self, led):
        board, sim = led
        for _ in range(3):
            with _answered_late(board, sim), pytest.raises(BoardNoReplyError):
                board.led_on(channel=3, mA=20)
            assert board.exchange_command('LED2_OFF') == 'LED 2 has been turned off'

    def test_rapid_led_on_off_every_write_confirms(self, led):
        """25 iterations of leds_off -> led_on -> leds_off, the characterization
        pattern that caused LED dropout. A write the board did not answer
        would raise."""
        board, _sim = led
        for _ in range(25):
            board.leds_off()
            board.led_on(channel=3, mA=20)
            board.leds_off()


@pytest.mark.firmware_tier
class TestTheMultilineRead:
    def test_stop_on_empty_ends_a_one_line_info_at_its_first_empty_line(self, motor_3_0):
        """The motor firmware sends INFO as one line; reading the rest of six
        lines to their per-line timeouts cost 2.5 s on every connect."""
        board, _sim = motor_3_0
        t0 = time.monotonic()
        resp = board.exchange_command('INFO', response_numlines=6, timeout=0.5, stop_on_empty=True)
        elapsed = time.monotonic() - t0
        assert 'v3.0.9' in resp[0]
        assert resp[1:] == [''] * 5
        assert elapsed < 1.5, f'the read waited out the empty lines: {elapsed:.2f}s'

    def test_stop_on_empty_reads_every_line_of_a_multiline_info(self, led):
        board, _sim = led
        stopped = board.exchange_command('INFO', response_numlines=6, stop_on_empty=True)
        whole = board.exchange_command('INFO', response_numlines=6)
        assert stopped == whole
        assert stopped[3] == 'Calibration:  Std'


# ---------------------------------------------------------------------------
# The motor board's moves, reads, parse and scale
# ---------------------------------------------------------------------------


@pytest.mark.firmware_tier
class TestTheMotorWrites:
    def test_a_move_writes_its_steps_as_the_target(self, motor):
        board, sim = motor
        sim.take_writes()
        board.move('Z', 1000)
        assert _writes_to(sim, 'Z', XTARGET) == [1000]

    def test_a_negative_move_reaches_the_chip_in_twos_complement(self, motor):
        board, sim = motor
        sim.take_writes()
        board.move('Z', -100)
        assert _writes_to(sim, 'Z', XTARGET) == [-100 + 0x100000000]

    def test_a_target_above_travel_is_driven_not_clamped(self, motor):
        """Travel is refused by the motion API before it reaches the driver; a
        driver that clamped made a refused move look like one that succeeded
        and stopped short."""
        board, sim = motor
        sim.take_writes()
        board.move_abs_pos('Z', 99999)
        assert _writes_to(sim, 'Z', XTARGET) == [board.z_um2ustep(99999)]

    def test_a_target_below_travel_is_driven_not_clamped(self, motor):
        board, sim = motor
        sim.take_writes()
        board.move_abs_pos('Z', -100)
        assert _writes_to(sim, 'Z', XTARGET) == [board.z_um2ustep(-100) + 0x100000000]

    def test_a_move_on_an_axis_the_board_lacks_is_refused(self, motor):
        board, _sim = motor
        with pytest.raises(PartAbsentError, match=r'Unsupported axis \(Q\)'):
            board.move_abs_pos('Q', 100)

    def test_spi_read_returns_the_registers_value(self, motor):
        board, _sim = motor
        board.move('X', 4321)
        assert board.spi_read('X', 0x20 + XACTUAL) == '4321'

    def test_target_status_is_true_once_the_move_has_arrived(self, motor):
        board, _sim = motor
        board.move_abs_pos('Z', 1000)
        assert board.target_status('Z') is True


@pytest.mark.firmware_tier
class TestAMoveUnderWay:
    @pytest.fixture
    def moving(self):
        """A board on realistic timing, a Z move to 5000 um just started."""
        board, _sim = motor_on_the_wire(timing='realistic')
        board.move_abs_pos('Z', 5000)
        return board

    def test_target_status_is_false_before_the_move_arrives(self, moving):
        assert moving.target_status('Z') is False

    def test_target_pos_reads_where_the_axis_is_going(self, moving):
        assert moving.target_pos('Z') == pytest.approx(5000, abs=0.01)
        assert moving.current_pos('Z') < 5000


@pytest.mark.firmware_tier
class TestTheMotorBoardsConfig:
    def test_get_axis_limits(self, motor):
        board, _sim = motor
        assert board.get_axis_limits('Z') == {'min': 0.0, 'max': 14000.0}

    def test_get_axis_limits_on_an_axis_the_board_lacks_is_refused(self, motor):
        board, _sim = motor
        with pytest.raises(PartAbsentError, match=r'Unsupported axis \(Q\)'):
            board.get_axis_limits('Q')

    def test_get_axes_config_has_all_four_axes_of_a_turret_scope(self, motor):
        board, _sim = motor
        assert set(board.get_axes_config()) == {'X', 'Y', 'Z', 'T'}

    def test_a_scope_with_no_turret_reads_as_one(self):
        board, _sim = motor_on_the_wire('LS850', 'XYZ')
        board.fullinfo()
        assert board.has_turret() is False


@pytest.mark.firmware_tier
class TestMotorBoardConversions:
    """The scale is the board's own CONFIG (170666 usteps/mm on Z, 20157 on
    XY, 80000 a turret position)."""

    def test_z_roundtrip(self, motor):
        board, _sim = motor
        for um in (0, 100, 5000, 14000):
            um_back = board.z_ustep2um(board.z_um2ustep(um))
            assert abs(um - um_back) < 0.01, f'Z roundtrip failed for {um}: got {um_back}'

    def test_xy_roundtrip(self, motor):
        board, _sim = motor
        for um in (0, 1000, 60000, 120000):
            um_back = board.xy_ustep2um(board.xy_um2ustep(um))
            assert abs(um - um_back) < 0.1, f'XY roundtrip failed for {um}: got {um_back}'

    def test_turret_roundtrip(self, motor):
        board, _sim = motor
        for pos in (1, 2, 3, 4):
            pos_back = board.t_ustep2pos(board.t_pos2ustep(pos))
            assert pos == pos_back, f'Turret roundtrip failed for pos {pos}: got {pos_back}'

    def test_z_known_values(self, motor):
        board, _sim = motor
        assert board.z_um2ustep(1000) == 170666
        assert abs(board.z_ustep2um(170666) - 1000) < 0.01

    def test_xy_known_values(self, motor):
        board, _sim = motor
        assert board.xy_um2ustep(1000) == 20157

    def test_turret_degrees(self, motor):
        """Position 1 = 0 deg, position 2 = 90 deg."""
        board, _sim = motor
        assert board.t_pos2ustep(1) == 0
        assert board.t_pos2ustep(2) == 80000


@pytest.mark.firmware_tier
class TestTheFirmwareVersion:
    def test_the_field_motor_firmware_has_no_version(self, motor):
        board, _sim = motor
        assert board.firmware_version is None
        assert board.is_v2 is False

    def test_the_3_0_motor_firmware_is_v2_or_later(self, motor_3_0):
        board, _sim = motor_3_0
        assert board.firmware_version == '3.0.9'
        assert board.is_v2 is True

    def test_the_field_led_firmware_has_no_version(self, led):
        board, _sim = led
        assert board.firmware_version is None
        assert board.is_v2 is False


# ---------------------------------------------------------------------------
# The port itself, on a scripted port (tests/scripted_port.py): the
# firmware plays no part in these claims
# ---------------------------------------------------------------------------

_ANSWERING = {'INFO': ['EL-0940 board'], 'PING': ['PONG']}
_WRITE_TIMEOUT = serial.SerialTimeoutException('Write timeout')
_PORT_LOST = serial.SerialException('write failed: [Errno 6] Device not configured')


def _connected(link, board_cls=SerialBoard):
    """The production driver, connected over a scripted port onto *link*."""
    if board_cls is SerialBoard:
        board = SerialBoard(LED_VID, LED_PID, '[Test]', backend=ScriptedBackend(link))
        board.connect()
    else:
        board = board_cls(backend=ScriptedBackend(link))
    assert board.is_connected()
    return board


def _lines(caplog, text):
    return [r for r in caplog.records if r.name == 'LVP.serial' and text in r.getMessage()]


class TestAWriteThatTimesOut:
    def test_it_is_no_reply_and_the_port_stays_open_for_the_next_command(self):
        link = ScriptedLink(_ANSWERING)
        board = _connected(link)
        link.write_fault = _WRITE_TIMEOUT
        with pytest.raises(BoardNoReplyError, match=r'^PING: the write to the board timed out'):
            board.exchange_command('PING')
        link.write_fault = None
        assert board.exchange_command('PING') == 'PONG'
        assert link.opens == 1, 'the timeout closed the port'

    def test_a_second_inside_the_log_window_is_not_logged_again(self, caplog):
        """#539: a link that keeps timing out must not flood serial.log."""
        board = _connected(link := ScriptedLink(_ANSWERING))
        link.write_fault = _WRITE_TIMEOUT
        with caplog.at_level(logging.WARNING, logger='LVP.serial'):
            for command in ('PING', 'PING'):
                with pytest.raises(BoardNoReplyError):
                    board.exchange_command(command)
        assert len(_lines(caplog, '-> TIMEOUT')) == 1

    def test_one_past_the_log_window_is_logged_again(self, monkeypatch, caplog):
        board = _connected(link := ScriptedLink(_ANSWERING))
        now = [100.0]
        monkeypatch.setattr(
            serialboard, 'time', SimpleNamespace(monotonic=lambda: now[0], sleep=time.sleep)
        )
        link.write_fault = _WRITE_TIMEOUT
        with caplog.at_level(logging.WARNING, logger='LVP.serial'):
            for at in (100.0, 103.0):
                now[0] = at
                with pytest.raises(BoardNoReplyError):
                    board.exchange_command('PING')
        assert len(_lines(caplog, '-> TIMEOUT')) == 2


class TestAPortThatFails:
    """A failed port is closed and stays closed until a connect; a board
    connected again whose port fails inside the log window is not logged
    twice (#539)."""

    def test_a_failure_before_the_send_closes_the_port_and_nothing_reopens_it(self):
        link = ScriptedLink(_ANSWERING)
        board = _connected(link)
        link.write_fault = _PORT_LOST
        with pytest.raises(BoardPortFailedError, match=r'^PING: the port failed'):
            board.exchange_command('PING')
        link.write_fault = None
        with pytest.raises(BoardPortFailedError, match=r'^PING: the port is closed'):
            board.exchange_command('PING')
        assert link.opens == 1

    def test_a_read_lost_after_the_send_is_no_reply_and_closes_the_port(self):
        """The board may have taken the command whose answer was lost."""
        link = ScriptedLink(_ANSWERING)
        board = _connected(link)
        link.read_fault = OSError('port disappeared')
        with pytest.raises(
            BoardNoReplyError, match=r'^FULLINFO: the port failed after the command was sent'
        ):
            board.exchange_multiline('FULLINFO', timeout=1)
        assert not board.is_connected()

    def test_an_exchange_failing_again_inside_the_log_window_is_not_logged_again(self, caplog):
        link = ScriptedLink(_ANSWERING)
        board = _connected(link)
        with caplog.at_level(logging.ERROR, logger='LVP.serial'):
            for _ in range(2):
                link.write_fault = _PORT_LOST
                with pytest.raises(BoardPortFailedError):
                    board.exchange_command('PING')
                link.write_fault = None
                board.connect()
        assert len(_lines(caplog, '-> EXCEPTION')) == 1

    def test_a_fast_write_failing_again_inside_the_log_window_is_not_logged_again(self, caplog):
        caplog.set_level(logging.DEBUG)
        link = ScriptedLink(_ANSWERING)
        board = _connected(link, LEDBoard)
        for _ in range(2):
            link.write_fault = _PORT_LOST
            with pytest.raises(BoardPortFailedError):
                board.leds_off_fast()
            link.write_fault = None
            board.connect()
        failed = [
            m
            for m in messages(records_of(caplog, serialboard))
            if '_write_command_fast(LEDS_OFF) failed' in m
        ]
        assert len(failed) == 1


class TestABoardThatSendsNothing:
    """#619: a board that sends no byte through the whole connect is hung
    firmware or a stuck USB hub, and needs a power cycle; a board that
    answers without a version is not silent
    (`test_sim_wire_led.py::TestTheFirmwareBoots`)."""

    @pytest.fixture(autouse=True)
    def _no_recovery_waits(self, monkeypatch):
        """The reset's waits (5 s after the soft reset) give a real board
        time to boot; the scripted board answers at once or never."""
        monkeypatch.setattr(
            serialboard, 'time', SimpleNamespace(monotonic=time.monotonic, sleep=lambda _: None)
        )

    @pytest.fixture
    def silent(self):
        link = ScriptedLink()
        return _connected(link), link

    def test_it_is_not_responsive_and_has_no_version(self, silent):
        board, _link = silent
        assert board.is_responsive() is False
        assert board.firmware_version is None

    def test_the_soft_reset_is_tried_before_it_is_called_silent(self, silent):
        """A board Thonny left in raw REPL looks silent until Ctrl-B and
        Ctrl-D wake it (seen 2026-04-14 on an LS850T), so the recovery runs
        before the verdict."""
        _board, link = silent
        assert b'\x02' in link.written and b'\x04' in link.written

    def test_every_command_is_refused_at_once_with_nothing_sent_or_reopened(self, silent):
        board, link = silent
        sent, opens = len(link.written), link.opens
        for _ in range(10):
            with pytest.raises(BoardNoReplyError, match='power cycle it'):
                board.exchange_command('LEDS_OFF')
        assert len(link.written) == sent
        assert link.opens == opens

    def test_info_still_reaches_it_and_no_line_at_all_is_no_reply(self, silent):
        """INFO is let through so a later connect can re-probe the board;
        the detection's own read, every line empty, is no reply."""
        board, link = silent
        sent = link.commands().count('INFO')
        with pytest.raises(BoardNoReplyError, match=r'^INFO: no reply from the board'):
            board.exchange_command('INFO', response_numlines=6, stop_on_empty=True)
        assert link.commands().count('INFO') == sent + 1

    def test_a_board_that_answers_again_is_not_silent_at_its_next_connect(self, silent):
        board, link = silent
        link.answers.update(_ANSWERING)
        board.connect()
        assert board.is_responsive() is True
        assert board.exchange_command('PING') == 'PONG'


# ---------------------------------------------------------------------------
# Left on scripted replies: an LED firmware that reports a version (the
# emulation carries only the field LED firmware, which reports none)
# ---------------------------------------------------------------------------


def _make_mock_serial(**overrides):
    """A mock serial.Serial that behaves like a connected port."""
    mock = MagicMock(spec=serial.Serial)
    mock.readline.return_value = b'OK\r\n'
    mock.write.return_value = None
    mock.close.return_value = None
    mock.in_waiting = 0
    mock.read.return_value = b''
    for k, v in overrides.items():
        setattr(mock, k, v)
    return mock


class TestLEDFirmwareVersion:
    """An LED firmware that reports a version: the emulation carries only
    the field LED firmware, which reports none."""

    def _make_board(self):
        board = LEDBoard.__new__(LEDBoard)
        board.found = False
        board._lock = threading.RLock()
        board._label = '[LED Class ]'
        board.port = '/dev/fake'
        board.baudrate = 115200
        board.bytesize = serial.EIGHTBITS
        board.parity = serial.PARITY_NONE
        board.stopbits = serial.STOPBITS_ONE
        board.timeout = 0.1
        board.write_timeout = 0.1
        board.driver = _make_mock_serial()
        board.firmware_version = None
        board.firmware_date = None
        board.firmware_responding = False
        board.protocol_version = None
        board._last_error_log_time = 0.0
        board._error_log_interval = 2.0
        return board

    def test_detect_v2_firmware(self):
        """Should parse v2.0.1 from INFO response."""
        board = self._make_board()
        # exchange_command('INFO', response_numlines=6) reads up to 6 lines;
        # first line is echo (RE: prefix), rest are content or empty (timeout).
        board.driver.readline.side_effect = [
            b'RE: INFO\r\n',
            b'Firmware:     2026-03-06 v2.0.1\r\n',
            b'\r\n',
            b'\r\n',
            b'\r\n',
            b'\r\n',
            b'\r\n',
            b'\r\n',
        ]
        board._detect_firmware_version()
        assert board.firmware_version == '2.0.1'
        assert board.is_v2 is True

    def test_detect_v2_0_0(self):
        """Should parse v2.0.0."""
        board = self._make_board()
        board.driver.readline.side_effect = [
            b'RE: INFO\r\n',
            b'Firmware:     2026-01-15 v2.0.0\r\n',
            b'\r\n',
            b'\r\n',
            b'\r\n',
            b'\r\n',
            b'\r\n',
            b'\r\n',
        ]
        board._detect_firmware_version()
        assert board.firmware_version == '2.0.0'
        assert board.is_v2 is True

    def test_detect_v1_firmware(self):
        """Hypothetical v1.x firmware should not be is_v2."""
        board = self._make_board()
        board.driver.readline.side_effect = [
            b'RE: INFO\r\n',
            b'Firmware: v1.5.0\r\n',
            b'\r\n',
            b'\r\n',
            b'\r\n',
            b'\r\n',
            b'\r\n',
            b'\r\n',
        ]
        board._detect_firmware_version()
        assert board.firmware_version == '1.5.0'
        assert board.is_v2 is False

    def test_detect_no_echo_firmware(self):
        """Future firmware without RE: echo should still parse version."""
        board = self._make_board()
        board.driver.readline.side_effect = [
            b'Firmware:     2027-01-01 v3.0.0\r\n',
            b'\r\n',
            b'\r\n',
            b'\r\n',
            b'\r\n',
            b'\r\n',
            b'\r\n',
        ]
        board._detect_firmware_version()
        assert board.firmware_version == '3.0.0'
        assert board.is_v2 is True


class TestSimulatorFirmwareVersion:
    """Test that simulators expose firmware_version and is_v2."""

    def test_led_simulator_default_v2(self):
        from drivers.simulated_ledboard import SimulatedLEDBoard

        board = SimulatedLEDBoard()
        assert board.firmware_version == '2.0.1'
        assert board.is_v2 is True

    def test_led_simulator_legacy(self):
        from drivers.simulated_ledboard import SimulatedLEDBoard

        board = SimulatedLEDBoard(firmware_version=None)
        assert board.firmware_version is None
        assert board.is_v2 is False

    def test_led_simulator_custom_version(self):
        from drivers.simulated_ledboard import SimulatedLEDBoard

        board = SimulatedLEDBoard(firmware_version='1.0.0')
        assert board.firmware_version == '1.0.0'
        assert board.is_v2 is False

    def test_motor_simulator_default_v2(self):
        from drivers.simulated_motorboard import SimulatedMotorBoard

        board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        assert board.firmware_version == '2.0.1'
        assert board.is_v2 is True

    def test_motor_simulator_legacy(self):
        from drivers.simulated_motorboard import SimulatedMotorBoard

        board = SimulatedMotorBoard(
            motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, firmware_version=None
        )
        assert board.firmware_version is None
        assert board.is_v2 is False


class TestCameraStateLock:
    """Verify Camera _state_lock protects active/device_removed flags."""

    def test_mark_disconnected_sets_device_removed_flag(self):
        """_mark_disconnected should set the device-removed flag under
        _state_lock. It must NOT clear _active -- that release was relocated
        to disconnect() on the async-teardown daemon thread by f01133c (bench
        D2 Windows-11 daA3840 fix). Releasing _active from the SDK callback
        thread fires the C++ wrapper destructor synchronously and races
        in-flight grab work. TestCameraMarkDisconnectedDoesNotReleaseActiveOnCallbackThread
        in test_audit_fixes.py guards the source-side contract; this is the
        runtime-side mirror.
        """
        from drivers.camera import Camera

        # Create a minimal concrete Camera subclass
        class StubCamera(Camera):
            def connect(self):
                self.active = True
                return True

            def _disconnect(self):
                self.active = None
                return True

            def is_connected(self):
                return self.active is not None and self.active is not False

            def init_camera_config(self):
                pass

            def _start_grabbing(self):
                pass

            def _stop_grabbing(self):
                pass

            def is_grabbing(self):
                return False

            def _frame_grid(self):
                return None

            def _set_hardware_window(self, plan):
                return True

            def get_min_frame_size(self):
                return {'w': 1, 'h': 1}

            def get_max_frame_size(self):
                return {'w': 4096, 'h': 4096}

            def _hardware_frame_size(self):
                return {'w': 1024, 'h': 1024}

            def set_pixel_format(self, f):
                return True

            def get_pixel_format(self):
                return 'Mono8'

            def get_supported_pixel_formats(self):
                return ('Mono8',)

            def exposure_t(self, exposure_ms):
                pass

            def get_exposure_t(self):
                return 10.0

            def auto_exposure_t(self, state=True):
                pass

            def get_all_temperatures(self):
                return {}

            def set_max_acquisition_frame_rate(self, enabled, fps=1.0):
                pass

            def _set_hardware_binning(self, size):
                return True

            def get_binning_size(self):
                return 1

            def grab_new_capture(self, timeout):
                return None

            def update_auto_gain_target_brightness(self, v):
                pass

            def update_auto_gain_min_max(self, min_g, max_g):
                pass

            def get_gain(self):
                return 1.0

            def gain(self, g):
                pass

            def auto_gain(self, state=True, **kw):
                pass

            def auto_gain_once(self, state=True, **kw):
                pass

            def set_test_pattern(self, enabled=False, pattern='Black'):
                pass

        cam = StubCamera()
        assert cam.active is True
        assert cam._device_removed is False

        cam._mark_disconnected()
        # _active is intentionally NOT cleared here -- the actual release
        # happens in disconnect() on a safe thread context. Asserting it
        # remains True after _mark_disconnected pins that contract; any
        # future refactor that re-introduces _active=None inside
        # _mark_disconnected will trip this guard alongside the AST guard.
        assert cam.active is True
        assert cam._device_removed is True
        assert cam.is_device_removed() is True

    def test_grab_returns_false_after_disconnect(self):
        """grab() should check flags under _state_lock and return False."""
        from drivers.camera import Camera

        class StubCamera(Camera):
            def connect(self):
                self.active = True
                return True

            def _disconnect(self):
                self.active = None
                return True

            def is_connected(self):
                return self.active is not None

            def init_camera_config(self):
                pass

            def _start_grabbing(self):
                pass

            def _stop_grabbing(self):
                pass

            def is_grabbing(self):
                return False

            def _frame_grid(self):
                return None

            def _set_hardware_window(self, plan):
                return True

            def get_min_frame_size(self):
                return {'w': 1, 'h': 1}

            def get_max_frame_size(self):
                return {'w': 4096, 'h': 4096}

            def _hardware_frame_size(self):
                return {'w': 1024, 'h': 1024}

            def set_pixel_format(self, f):
                return True

            def get_pixel_format(self):
                return 'Mono8'

            def get_supported_pixel_formats(self):
                return ('Mono8',)

            def exposure_t(self, exposure_ms):
                pass

            def get_exposure_t(self):
                return 10.0

            def auto_exposure_t(self, state=True):
                pass

            def get_all_temperatures(self):
                return {}

            def set_max_acquisition_frame_rate(self, enabled, fps=1.0):
                pass

            def _set_hardware_binning(self, size):
                return True

            def get_binning_size(self):
                return 1

            def grab_new_capture(self, timeout):
                return None

            def update_auto_gain_target_brightness(self, v):
                pass

            def update_auto_gain_min_max(self, min_g, max_g):
                pass

            def get_gain(self):
                return 1.0

            def gain(self, g):
                pass

            def auto_gain(self, state=True, **kw):
                pass

            def auto_gain_once(self, state=True, **kw):
                pass

            def set_test_pattern(self, enabled=False, pattern='Black'):
                pass

        cam = StubCamera()
        assert cam.active is True

        # Normal grab without image handler answers no frame (no handler)
        assert cam.grab() is None

        # After disconnect, grab answers no frame immediately
        cam._mark_disconnected()
        assert cam.grab() is None

    def test_concurrent_mark_disconnected(self):
        """Multiple threads calling _mark_disconnected should not raise."""
        from drivers.camera import Camera

        class StubCamera(Camera):
            def connect(self):
                self.active = True
                return True

            def _disconnect(self):
                return True

            def is_connected(self):
                return self.active is not None

            def init_camera_config(self):
                pass

            def _start_grabbing(self):
                pass

            def _stop_grabbing(self):
                pass

            def is_grabbing(self):
                return False

            def _frame_grid(self):
                return None

            def _set_hardware_window(self, plan):
                return True

            def get_min_frame_size(self):
                return {'w': 1, 'h': 1}

            def get_max_frame_size(self):
                return {'w': 4096, 'h': 4096}

            def _hardware_frame_size(self):
                return {'w': 1024, 'h': 1024}

            def set_pixel_format(self, f):
                return True

            def get_pixel_format(self):
                return 'Mono8'

            def get_supported_pixel_formats(self):
                return ('Mono8',)

            def exposure_t(self, exposure_ms):
                pass

            def get_exposure_t(self):
                return 10.0

            def auto_exposure_t(self, state=True):
                pass

            def get_all_temperatures(self):
                return {}

            def set_max_acquisition_frame_rate(self, enabled, fps=1.0):
                pass

            def _set_hardware_binning(self, size):
                return True

            def get_binning_size(self):
                return 1

            def grab_new_capture(self, timeout):
                return None

            def update_auto_gain_target_brightness(self, v):
                pass

            def update_auto_gain_min_max(self, min_g, max_g):
                pass

            def get_gain(self):
                return 1.0

            def gain(self, g):
                pass

            def auto_gain(self, state=True, **kw):
                pass

            def auto_gain_once(self, state=True, **kw):
                pass

            def set_test_pattern(self, enabled=False, pattern='Black'):
                pass

        cam = StubCamera()
        errors = []

        def disconnect_loop():
            try:
                for _ in range(50):
                    cam._mark_disconnected()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=disconnect_loop) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert not errors
        assert cam._device_removed is True
        # _active intentionally NOT cleared by _mark_disconnected (see
        # test_mark_disconnected_sets_device_removed_flag above for the
        # callback-thread-safety rationale, f01133c). disconnect() owns
        # the release.
        assert cam.active is True
