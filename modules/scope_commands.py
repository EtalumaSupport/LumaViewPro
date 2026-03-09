# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
GUI-independent scope hardware command functions.

These functions wrap scope operations with executor-based dispatch,
taking scope and executor as explicit parameters. They can be used
by LumaViewPro, the REST API, or standalone scripts.

Optional callbacks allow the caller to wire up GUI updates or other
side effects without coupling this module to any UI framework.
"""

from lvp_logger import logger
from modules.sequential_io_executor import IOTask


# ---------------------------------------------------------------------------
# LED commands
# ---------------------------------------------------------------------------

def leds_off(scope, executor, callback=None):
    """Turn off all LEDs via the executor.

    Args:
        scope: Lumascope instance with .led and .leds_off()
        executor: SequentialIOExecutor for serial command dispatch
        callback: Optional callback invoked after LEDs are turned off
    """
    if not scope.led:
        logger.warning('[ScopeCmd  ] LED controller not available.')
        return

    executor.put(IOTask(action=scope.leds_off, callback=callback))
    logger.info('[ScopeCmd  ] scope.leds_off()')


def led_on(scope, executor, channel, illumination, callback=None, cb_kwargs=None):
    """Turn on a specific LED channel via the executor.

    Args:
        scope: Lumascope instance with .led and .led_on()
        executor: SequentialIOExecutor for serial command dispatch
        channel: LED channel number
        illumination: Illumination level (mA)
        callback: Optional callback invoked after LED is turned on
        cb_kwargs: Optional keyword arguments for callback
    """
    if not scope.led:
        logger.warning('[ScopeCmd  ] LED controller not available.')
        return

    executor.put(IOTask(
        action=scope.led_on,
        args=(channel, illumination),
        callback=callback,
        cb_kwargs=cb_kwargs,
    ))


def led_off(scope, executor, channel, callback=None, cb_kwargs=None):
    """Turn off a specific LED channel via the executor.

    Args:
        scope: Lumascope instance with .led and .led_off()
        executor: SequentialIOExecutor for serial command dispatch
        channel: LED channel number
        callback: Optional callback invoked after LED is turned off
        cb_kwargs: Optional keyword arguments for callback
    """
    if not scope.led:
        logger.warning('[ScopeCmd  ] LED controller not available.')
        return

    executor.put(IOTask(
        action=scope.led_off,
        kwargs={'channel': channel},
        callback=callback,
        cb_kwargs=cb_kwargs,
    ))


def led_on_sync(scope, executor, channel, illumination, timeout=5):
    """Turn on a specific LED channel synchronously (blocks until complete).

    Args:
        scope: Lumascope instance
        executor: SequentialIOExecutor
        channel: LED channel number
        illumination: Illumination level (mA)
        timeout: Seconds to wait for completion
    """
    if not scope.led:
        logger.warning('[ScopeCmd  ] LED controller not available.')
        return

    task = IOTask(action=scope.led_on, args=(channel, illumination))
    fut = executor.put(task, return_future=True)
    if fut:
        fut.result(timeout=timeout)


# ---------------------------------------------------------------------------
# Motion commands
# ---------------------------------------------------------------------------

def move_absolute(
    scope,
    executor,
    axis,
    pos,
    wait_until_complete=False,
    overshoot_enabled=True,
    callback=None,
    cb_kwargs=None,
):
    """Move to an absolute position via the executor.

    Args:
        scope: Lumascope instance with .move_absolute_position()
        executor: SequentialIOExecutor for serial command dispatch
        axis: Axis letter ('X', 'Y', 'Z')
        pos: Target position in micrometers
        wait_until_complete: Block until motion finishes
        overshoot_enabled: Allow overshoot compensation
        callback: Optional callback invoked after move completes
        cb_kwargs: Optional keyword arguments for callback
    """
    executor.put(IOTask(
        action=scope.move_absolute_position,
        kwargs={
            'axis': axis,
            'pos': pos,
            'wait_until_complete': wait_until_complete,
            'overshoot_enabled': overshoot_enabled,
        },
        callback=callback,
        cb_kwargs=cb_kwargs,
    ))


def move_relative(
    scope,
    executor,
    axis,
    um,
    wait_until_complete=False,
    overshoot_enabled=True,
    callback=None,
    cb_kwargs=None,
):
    """Move by a relative offset via the executor.

    Args:
        scope: Lumascope instance with .move_relative_position()
        executor: SequentialIOExecutor for serial command dispatch
        axis: Axis letter ('X', 'Y', 'Z')
        um: Distance in micrometers
        wait_until_complete: Block until motion finishes
        overshoot_enabled: Allow overshoot compensation
        callback: Optional callback invoked after move completes
        cb_kwargs: Optional keyword arguments for callback
    """
    executor.put(IOTask(
        action=scope.move_relative_position,
        kwargs={
            'axis': axis,
            'um': um,
            'wait_until_complete': wait_until_complete,
            'overshoot_enabled': overshoot_enabled,
        },
        callback=callback,
        cb_kwargs=cb_kwargs,
    ))


def move_home(scope, executor, axis, callback=None, cb_args=None):
    """Home an axis via the executor.

    Args:
        scope: Lumascope instance with .zhome(), .xyhome(), .thome()
        executor: SequentialIOExecutor for serial command dispatch
        axis: Axis string ('Z', 'XY', 'T')
        callback: Optional callback invoked after homing completes
        cb_args: Optional positional arguments for callback
    """
    axis = axis.upper()
    if axis == 'Z':
        executor.put(IOTask(action=scope.zhome, callback=callback, cb_args=cb_args))
    elif axis == 'XY':
        executor.put(IOTask(action=scope.xyhome, callback=callback, cb_args=cb_args))
    elif axis == 'T':
        executor.put(IOTask(action=scope.thome, callback=callback, cb_args=cb_args))
    else:
        logger.warning(f'[ScopeCmd  ] Unknown home axis: {axis}')
