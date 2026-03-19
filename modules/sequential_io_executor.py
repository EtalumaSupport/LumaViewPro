# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
try:
    from kivy.clock import Clock
except ImportError:
    # Subprocess mode - dummy Clock
    class Clock:
        @staticmethod
        def schedule_once(func, timeout): 
            # In subprocess, call immediately without UI scheduling
            if callable(func):
                try:
                    func(0)  # Call with dummy dt=0
                except Exception:
                    pass
        
        @staticmethod
        def schedule_interval(func, interval): 
            return None
        
        @staticmethod
        def unschedule(event): 
            pass

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, CancelledError
import queue
from collections.abc import Sequence
from functools import partial
from lvp_logger import logger, debug
import threading
import time


"""
IOTask
- Encapsulates a single unit of work:
    • action:         callable performing the I/O
    • args, kwargs:   parameters for action
    • callback:       optional function to call when done
    • cb_args, cb_kwargs: arguments for callback
    • pass_result:    if True, injects (result, exception) into cb_kwargs
- Usage:
    task = IOTask(
        action=grab_image,
        args=(well_id,),
        kwargs={'exposure_ms':100},
        callback=display_result,
        pass_result=True
    )
    executor.enqueue(task)
"""
class IOTask:
        def __init__(self, action, args=None, kwargs=None, callback=None, cb_args=None, cb_kwargs=None, pass_result=False):
            self.action = action
            if args is None:
                self.args = ()
            # if it’s a sequence (list, tuple, etc) but not a string
            elif isinstance(args, Sequence) and not isinstance(args, (str, bytes)):
                self.args = tuple(args)
            else:
                self.args = (args,)

            self.kwargs = kwargs if kwargs is not None else {}
            self.callback = callback
            self.protocol = None
            self.name = ""
            
            
            if cb_args is None:
                self.cb_args = ()
            # if it’s a sequence (list, tuple, etc) but not a string
            elif isinstance(cb_args, Sequence) and not isinstance(cb_args, (str, bytes)):
                self.cb_args = tuple(cb_args)
            else:
                self.cb_args = (cb_args,)

            self.cb_kwargs = cb_kwargs if cb_kwargs is not None else {}
            self.pass_result = pass_result

        def run(self):
            try:
                threading.current_thread().name = self.name
                if not callable(self.action):
                    logger.warning(f"{self.name} Worker received non-callable action: {str(self.action)}")
                t_start = time.monotonic()
                res = self.action(*self.args, **self.kwargs)
                elapsed = time.monotonic() - t_start
                if elapsed > 5.0:
                    action_name = getattr(self.action, '__name__', str(self.action))
                    logger.warning(f"[IOTask    ] Slow task ({elapsed:.1f}s): {action_name} on {self.name}")
                return res, None
            except Exception as e:
                logger.error(f"Uncaught Thread Exception in {self.name} Worker: {e}", exc_info=True)
                return None, e

        def set_callback(self, callback, cb_args, cb_kwargs):
            self.callback = callback
            self.cb_args = cb_args
            self.cb_kwargs = cb_kwargs if cb_kwargs is not None else {}

        def on_complete(self, result, exception):
            if self.callback is None:
                return

            def _safe_callback(dt):
                try:
                    self.callback(*cb_args, **cb_kwargs)
                except Exception:
                    logger.error(f"[IOTask    ] Callback {self.callback} raised exception", exc_info=True)

            if self.pass_result:
                # Only copy when we need to mutate
                cb_kwargs = dict(self.cb_kwargs)
                cb_kwargs['result'] = result
                cb_kwargs['exception'] = exception
            else:
                cb_kwargs = self.cb_kwargs

            cb_args = self.cb_args
            Clock.schedule_once(_safe_callback, 0)

        def set_name(self, name):
            self.name = name

        def __call__(self):
            return self.run()
        
        def __repr__(self):
            return f"<IOTask: Action: {str(self.action)} Callback: {str(self.callback)}>"


"""
SequentialIOExecutor
- Manages a FIFO queue of IOTask instances.
- Uses a ThreadPoolExecutor (configurable max_workers) to run tasks in the background.
- Dispatches tasks one by one (or up to max_workers in parallel) and:
    1. Calls task.run() on a worker thread.
    2. Captures (result, exception).
    3. Schedules task.on_complete(result, exception) on the main/UI thread.
- Usage:
    executor = SequentialIOExecutor(max_workers=2)
    executor.start()
    executor.enqueue(task)
    # ... later ...
    executor.shutdown(wait=True)
"""
class SequentialIOExecutor:
    def __init__(self, max_workers: int=1, name: str=None):
        self.queue = queue.Queue()
        self.protocol_queue = queue.Queue()
        self.protocol_running = threading.Event()
        self.protocol_finish = threading.Event()
        self.name = name
        if name is not None:
            self.executor_name = name + "_" + "WORKER"
            self.dispatcher_name = name + "_" + "DISPATCHER"
            self.executor = ThreadPoolExecutor(thread_name_prefix=self.executor_name, max_workers=max_workers)
            self.dispatcher = ThreadPoolExecutor(thread_name_prefix=self.dispatcher_name, max_workers=1)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            self.dispatcher = ThreadPoolExecutor(max_workers=1)
        self._running_task_lock = threading.Lock()
        self._running_task = None
        self.global_callback = None
        self.pending_shutdown = False
        self.caller_futures = {}

        self.cleared_queue = False
        self.cleared_protocol_queue = False

        self._disable = False

        self.blocker = threading.Event()
        self.last_task_done_monotonic = time.monotonic()

        # Protocol completion callback support
        self.protocol_complete_callback = None
        self.protocol_complete_cb_args = ()
        self.protocol_complete_cb_kwargs = {}

    @property
    def running_task(self):
        with self._running_task_lock:
            return self._running_task

    @running_task.setter
    def running_task(self, value):
        with self._running_task_lock:
            self._running_task = value


    def start(self):
        # Start internal dispatcher
        self.dispatcher.submit(self._dispatch_loop)

    def disable(self):
        self._disable = True

    def enable(self):
        self._disable = False
        self.blocker.set()

    def put(self, task: IOTask, return_future: bool = False):
        if self._disable:
            return None

        if self.protocol_running.is_set() and not self.protocol_finish.is_set():
            return None
        
        # Push IO work item into queue
        # Only create Future if caller explicitly requests it to reduce memory overhead
        if return_future:
            fut = Future()
            self.caller_futures[task] = fut
        else:
            fut = None
        self.queue.put(task)
        task.set_name(self.executor_name)
        return fut

    def protocol_put(self, task: IOTask, return_future: bool = False):
        """
        Adds an IOTask to the Protocol Execution Queue
        NOTE: Protocol Execution Queue only executes when protocol is in session:
        ie protocol_start has been called.
        """
        if self._disable:
            return None
        
        if not self.protocol_running.is_set():
            return None
        
        # Only create Future if caller explicitly requests it to reduce memory overhead
        if return_future:
            fut = Future()
            self.caller_futures[task] = fut
        else:
            fut = None
        self.protocol_queue.put(task)
        task.set_name(self.executor_name)
        return fut

    def protocol_start(self):

        self.protocol_running.set()
        logger.info(f"{self.name} Protocol Started")

    def protocol_end(self):
        was_running = self.protocol_running.is_set()
        self.protocol_running.clear()
        # Clear completion callback when protocol ends prematurely
        self.protocol_complete_callback = None
        self.protocol_complete_cb_args = ()
        self.protocol_complete_cb_kwargs = {}
        if was_running:
            logger.info(f"{self.name} Protocol Ended")

    def protocol_finish_then_end(self):
        self.protocol_finish.set()
        logger.info(f"{self.name} set to complete protocol then end")

    def is_protocol_running(self):
        return self.protocol_running.is_set()

    def set_protocol_complete_callback(self, callback, cb_args=None, cb_kwargs=None):
        """Register callback to be invoked when protocol queue is fully drained."""
        self.protocol_complete_callback = callback
        self.protocol_complete_cb_args = cb_args if cb_args is not None else ()
        self.protocol_complete_cb_kwargs = cb_kwargs if cb_kwargs is not None else {}

    def is_protocol_queue_active(self) -> bool:
        """Returns True if protocol queue has pending tasks or is draining."""
        return (self.protocol_finish.is_set() or
                not self.protocol_queue.empty() or
                (self.running_task is not None and getattr(self.running_task, 'protocol', False)))

    def wait_for_task(self, task: IOTask, timeout: float):
        if task not in self.caller_futures:
            return
        
        try:
            fut: Future = self.caller_futures[task]
            result = fut.result(timeout=timeout)
        except Exception as e:
            logger.error(f"{self.name} Worker Error: {e}")
            

    def _dispatch_loop(self):
        # Pulls from queue, submits to worker pool, wires up callbacks
        threading.current_thread().name = self.dispatcher_name
        while True:
            if self._disable:
                self.blocker.wait()
            try:
                try:
                    if self.protocol_running.is_set() or self.protocol_finish.is_set():
                        task = self.protocol_queue.get(block=True, timeout=0.2)
                        task.protocol = True
                    elif not self.protocol_queue.empty():
                        # Protocol is not running and there are still items in the protocol queue
                        # Clear the queue
                        self.clear_protocol_pending()
                        continue
                    else:
                        task = self.queue.get(block=True, timeout=0.2)
                        task.protocol = False
                except queue.Empty:
                    if self.pending_shutdown:
                        return
                    if self.protocol_finish.is_set():
                        self.protocol_end()
                        self.protocol_finish.clear()
                        # Trigger completion callback if registered
                        if self.protocol_complete_callback is not None:
                            Clock.schedule_once(
                                lambda dt: self.protocol_complete_callback(
                                    *self.protocol_complete_cb_args,
                                    **self.protocol_complete_cb_kwargs
                                ),
                                0
                            )
                            # Clear callback after invoking
                            self.protocol_complete_callback = None
                            self.protocol_complete_cb_args = ()
                            self.protocol_complete_cb_kwargs = {}
                    continue
                if self.protocol_running.is_set() or self.protocol_finish.is_set():
                    if self.pending_shutdown:
                        return
                    # Inline to avoid holding future reference - GC can collect immediately after callback attached
                    self.executor.submit(task.run).add_done_callback(partial(self._safe_done_cb, task=task))
                    self.running_task = task
                else:
                    if not self.protocol_queue.empty():
                        self.protocol_queue.queue.clear()
                    if self.pending_shutdown:
                        return
                    # Inline to avoid holding future reference - GC can collect immediately after callback attached
                    self.executor.submit(task.run).add_done_callback(partial(self._safe_done_cb, task=task))
                    self.running_task = task
            except Exception as e:
                logger.error(f"Uncaught Thread Exception in {self.name} Dispatcher: {e}", exc_info=True)

    def _safe_done_cb(self, fut, task):
        try:
            if fut.cancelled():
                # Treat cancellation as a completed task with a CancelledError
                self._on_task_done(task, None, CancelledError())
                return

            exc = fut.exception()
            if exc is not None:
                # This would only happen if task.run() itself raised and wasn't caught,
                self._on_task_done(task, None, exc)
                return

            result = fut.result() 
            # task.run() returns (res, None) or (None, e)
            if isinstance(result, tuple) and len(result) == 2:
                self._on_task_done(task, result[0], result[1])
            else:
                # Backstop in case run() changes
                self._on_task_done(task, result, None)
        except Exception as e:
            logger.error(f"Done-callback error in {self.name}: {e}")
        finally:
            del fut


    def _on_task_done(self, task: IOTask, result, exception):
        # Receives (result, exception) from worker, then schedules task.on_complete
        self.last_task_done_monotonic = time.monotonic()
        caller_fut = self.caller_futures.pop(task, None)
        if caller_fut:
            # This future was returned to a caller - they still hold a reference
            # DON'T null internal state or it will break their .result() call
            if exception:
                caller_fut.set_exception(exception)
            else:
                caller_fut.set_result(result)
            # Only delete our local reference, not the object internals
            del caller_fut

        task.on_complete(result, exception)
        if task.protocol:
            if not self.cleared_protocol_queue:
                self.protocol_queue.task_done()
            else:
                self.clear_protocol_pending()
                self.cleared_protocol_queue = False
        else:
            if not self.cleared_queue:
                self.queue.task_done()
            else:
                self.clear_pending()
                self.cleared_queue = False

        self.running_task = None
        if self.global_callback is not None:
            # Lambda wrapper to consume dt parameter that Clock always passes
            Clock.schedule_once(lambda dt: self.global_callback(*self.global_cb_args, **self.global_cb_kwargs), 0)

    def set_done_callback(self, callback_fn, cb_args, cb_kwargs):
        # Allows to set a callback for when any IO task finishes (universal)
        self.global_callback = callback_fn
        self.global_cb_args = cb_args
        self.global_cb_kwargs = cb_kwargs

    def shutdown(self, wait=True):
        # Stops dispatcher and running tasks
        # If wait, wait until task running finishes
        self.pending_shutdown = True
        self.enable()
        self.protocol_end()
        self.clear_pending()
        self.clear_protocol_pending()
        self.dispatcher.shutdown(wait=wait, cancel_futures=not wait)
        self.executor.shutdown(wait=wait, cancel_futures=not wait)
        
        # Explicitly clear callback references and futures dict to break circular refs
        self.global_callback = None
        self.global_cb_args = None
        self.global_cb_kwargs = None
        
        # Clear futures dict - don't corrupt internals as callers may hold references
        # Just remove our tracking references
        self.caller_futures.clear()
        self.running_task = None

    def join(self, timeout=None):
        # Block until all queued tasks processed (or until timeout)
        pass

    def clear_pending(self):
        # Remove all tasks still in queue
        cleared_count = 0
        while True:
            try:
                task = self.queue.get_nowait()
                # Cancel future and aggressively cleanup
                fut = self.caller_futures.pop(task, None)
                if fut:
                    try:
                        fut.cancel()
                    except Exception:
                        pass
                cleared_count += 1
                # Balance out get_nowait with a task_done
                self.queue.task_done()
            except queue.Empty:
                break
        
        self.cleared_queue = True
        if cleared_count > 0:
            logger.info(f"{self.name} Pending Queue Cleared ({cleared_count} tasks)")

    def clear_protocol_pending(self):
        cleared_count = 0
        while True:
            try:
                task = self.protocol_queue.get_nowait()
                # Cancel future and aggressively cleanup
                fut = self.caller_futures.pop(task, None)
                if fut:
                    try:
                        fut.cancel()
                    except Exception:
                        pass
                cleared_count += 1
                # Balance out get_nowait with a task_done
                self.protocol_queue.task_done()
            except queue.Empty:
                break
        
        self.cleared_protocol_queue = True
        if cleared_count > 0:
            logger.info(f"{self.name} Pending Protocol Queue Cleared ({cleared_count} tasks)")
    
    def is_busy(self):
        # Returns true if tasks queued or running
        return not (self.queue.empty() and self.running_task is None)

    def queue_size(self) -> int:
        return self.queue.qsize()

    def protocol_queue_size(self) -> int:
        """Returns the number of pending protocol tasks, including any currently running task."""
        queue_count = self.protocol_queue.qsize()
        # Add 1 if there's a currently running protocol task
        if self.running_task is not None and getattr(self.running_task, 'protocol', False):
            queue_count += 1
        return queue_count

    def seconds_since_last_task(self) -> float:
        return time.monotonic() - self.last_task_done_monotonic



