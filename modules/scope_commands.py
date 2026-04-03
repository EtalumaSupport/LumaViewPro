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
    if not scope.led_connected:
        logger.warning('[ScopeCmd  ] LED controller not available.')
        return

    executor.put(IOTask(action=scope.leds_off, callback=callback))
    logger.info('[ScopeCmd  ] scope.leds_off()')


def led_on(scope, executor, channel, illumination, callback=None,
           cb_kwargs=None, owner: str = ''):
    """Turn on a specific LED channel via the executor.

    Args:
        scope: Lumascope instance with .led and .led_on()
        executor: SequentialIOExecutor for serial command dispatch
        channel: LED channel number
        illumination: Illumination level (mA)
        callback: Optional callback invoked after LED is turned on
        cb_kwargs: Optional keyword arguments for callback
        owner: Ownership tag passed through to scope.led_on()
    """
    if not scope.led_connected:
        logger.warning('[ScopeCmd  ] LED controller not available.')
        return

    kwargs = {'owner': owner} if owner else {}
    executor.put(IOTask(
        action=scope.led_on,
        args=(channel, illumination),
        kwargs=kwargs,
        callback=callback,
        cb_kwargs=cb_kwargs,
    ))


def led_off(scope, executor, channel, callback=None, cb_kwargs=None,
            owner: str = ''):
    """Turn off a specific LED channel via the executor.

    Args:
        scope: Lumascope instance with .led and .led_off()
        executor: SequentialIOExecutor for serial command dispatch
        channel: LED channel number
        callback: Optional callback invoked after LED is turned off
        cb_kwargs: Optional keyword arguments for callback
        owner: Ownership tag passed through to scope.led_off()
    """
    if not scope.led_connected:
        logger.warning('[ScopeCmd  ] LED controller not available.')
        return

    kwargs = {'channel': channel}
    if owner:
        kwargs['owner'] = owner
    executor.put(IOTask(
        action=scope.led_off,
        kwargs=kwargs,
        callback=callback,
        cb_kwargs=cb_kwargs,
    ))


def led_on_sync(scope, executor, channel, illumination, timeout=5,
                owner: str = ''):
    """Turn on a specific LED channel synchronously (blocks until complete).

    Args:
        scope: Lumascope instance
        executor: SequentialIOExecutor
        channel: LED channel number
        illumination: Illumination level (mA)
        timeout: Seconds to wait for completion
        owner: Ownership tag passed through to scope.led_on()
    """
    if not scope.led_connected:
        logger.warning('[ScopeCmd  ] LED controller not available.')
        return

    kwargs = {'owner': owner} if owner else {}
    task = IOTask(action=scope.led_on, args=(channel, illumination),
                  kwargs=kwargs)
    fut = executor.put(task, return_future=True)
    if fut:
        fut.result(timeout=timeout)


def leds_off_sync(scope, executor, timeout=5):
    """Turn off all LEDs synchronously (blocks until complete).

    Args:
        scope: Lumascope instance
        executor: SequentialIOExecutor
        timeout: Seconds to wait for completion
    """
    if not scope.led_connected:
        logger.warning('[ScopeCmd  ] LED controller not available.')
        return

    task = IOTask(action=scope.leds_off)
    fut = executor.put(task, return_future=True)
    if fut:
        fut.result(timeout=timeout)


# ---------------------------------------------------------------------------
# Camera commands
# ---------------------------------------------------------------------------

def set_gain_sync(scope, executor, gain, timeout=5):
    """Set camera gain synchronously via the executor.

    Args:
        scope: Lumascope instance with .set_gain()
        executor: SequentialIOExecutor (camera executor)
        gain: Gain value to set
        timeout: Seconds to wait for completion
    """
    task = IOTask(action=scope.set_gain, args=(gain,))
    fut = executor.put(task, return_future=True)
    if fut:
        fut.result(timeout=timeout)


def set_exposure_sync(scope, executor, exposure, timeout=5):
    """Set camera exposure time synchronously via the executor.

    Args:
        scope: Lumascope instance with .set_exposure_time()
        executor: SequentialIOExecutor (camera executor)
        exposure: Exposure time in milliseconds
        timeout: Seconds to wait for completion
    """
    task = IOTask(action=scope.set_exposure_time, args=(exposure,))
    fut = executor.put(task, return_future=True)
    if fut:
        fut.result(timeout=timeout)


def capture_and_wait_sync(scope, executor, timeout=30, **kwargs):
    """Capture a frame synchronously via the executor.

    Routes scope.capture_and_wait() through the camera executor so that
    all camera operations are serialized.

    Args:
        scope: Lumascope instance with .capture_and_wait()
        executor: SequentialIOExecutor (camera executor)
        timeout: Seconds to wait for the capture to complete
        **kwargs: Passed through to scope.capture_and_wait()

    Returns:
        The captured image array, or None on failure.
    """
    task = IOTask(action=scope.capture_and_wait, kwargs=kwargs)
    fut = executor.put(task, return_future=True)
    if fut:
        return fut.result(timeout=timeout)
    return None


# ---------------------------------------------------------------------------
# Motion commands
# ---------------------------------------------------------------------------

def move_absolute_sync(scope, executor, axis, pos, wait_until_complete=True,
                       overshoot_enabled=True, timeout=30):
    """Move to an absolute position synchronously via the executor.

    Blocks until the IOTask completes.  When *wait_until_complete* is True
    (the default), the executor worker thread also waits for the stage to
    arrive before returning — so the caller is guaranteed motion is finished.

    Args:
        scope: Lumascope instance
        executor: SequentialIOExecutor (io/stage executor)
        axis: Axis letter ('X', 'Y', 'Z')
        pos: Target position in micrometers
        wait_until_complete: Block inside the executor until motion finishes
        overshoot_enabled: Allow overshoot compensation
        timeout: Seconds to wait for the future
    """
    task = IOTask(
        action=scope.move_absolute_position,
        kwargs={
            'axis': axis,
            'pos': pos,
            'wait_until_complete': wait_until_complete,
            'overshoot_enabled': overshoot_enabled,
        },
    )
    fut = executor.put(task, return_future=True)
    if fut:
        fut.result(timeout=timeout)


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
