
from kivy.clock import Clock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import queue
from collections.abc import Sequence
from lvp_logger import logger
import traceback
import threading


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
        def __init__(self, action, args=None, kwargs={}, callback=None, cb_args=None, cb_kwargs={}, pass_result=False):
            self.action = action
            if args is None:
                self.args = ()
            # if it’s a sequence (list, tuple, etc) but not a string
            elif isinstance(args, Sequence) and not isinstance(args, (str, bytes)):
                self.args = tuple(args)
            else:
                self.args = (args,)

            self.kwargs = kwargs
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

            self.cb_kwargs = cb_kwargs.copy()
            self.pass_result = pass_result

        def run(self):
            try:
                res = self.action(*self.args, **self.kwargs)
                return res, None
            except Exception as e:
                logger.error(f"Uncaught Thread Exception in {self.name} Worker: {e}")
                traceback.print_exc()
                return None, e

        def set_callback(self, callback, cb_args, cb_kwargs):
            self.callback = callback
            self.cb_args = cb_args
            self.cb_kwargs = cb_kwargs.copy()

        def on_complete(self, result, exception):
            final_kwargs = dict(self.cb_kwargs)
            if self.pass_result:
                final_kwargs['result'] = result
                final_kwargs['exception'] = exception

            if self.callback is not None:
                Clock.schedule_once(lambda dt: self.callback(*self.cb_args, **final_kwargs), 0)

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
        self.protocol_running = False
        self.name = name
        if name is not None:
            executor_name = name + "_" + "WORKER"
            dispatcher_name = name + "_" + "DISPATCHER"
            self.executor = ThreadPoolExecutor(thread_name_prefix=executor_name, max_workers=max_workers)
            self.dispatcher = ThreadPoolExecutor(thread_name_prefix=dispatcher_name, max_workers=1)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            self.dispatcher = ThreadPoolExecutor(max_workers=1)
        self.running_task = None
        self.global_callback = None
        self.pending_shutdown = False
        self.caller_futures = {}

        self.executed_protocol_tasks = []
        self.executed_tasks = []
        self.cleared_queue = False
        self.cleared_protocol_queue = False

        self.queue_lock = threading.Lock()
        self.protocol_queue_lock = threading.Lock()


    def start(self):
        # Start internal dispatcher
        self.dispatcher.submit(self._dispatch_loop)

    # TODO: Have it return a future to be able to block until that thread has finished its task
    def put(self, task: IOTask):
        # Push IO work item into queue
        fut = Future()
        self.caller_futures[task] = fut
        self.queue.put(task)
        task.set_name(self.name)
        return fut

    def protocol_put(self, task: IOTask):
        fut = Future()
        self.caller_futures[task] = fut
        self.protocol_queue.put(task)
        task.set_name(self.name)
        return fut

    def protocol_start(self):
        self.protocol_running = True

    def protocol_end(self):
        self.protocol_running = False

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
        while True:
            try:
                try:
                    if self.protocol_running or not self.protocol_queue.empty():
                        task = self.protocol_queue.get(block=True, timeout=0.2)
                        task.protocol = True
                    else:
                        task = self.queue.get(block=True, timeout=0.2)
                        task.protocol = False
                except queue.Empty:
                    if self.pending_shutdown:
                        return
                    continue
                if self.protocol_running:
                    future = self.executor.submit(task.run)
                    future.add_done_callback(lambda fut, t=task: self._on_task_done(t, *fut.result()))
                    self.running_task = task
                    self.executed_protocol_tasks.append(task)
                else:
                    if self.protocol_queue.not_empty:
                        self.protocol_queue.queue.clear()
                    future = self.executor.submit(task.run)
                    future.add_done_callback(lambda fut, t=task: self._on_task_done(t, *fut.result()))
                    self.running_task = task
                    self.executed_tasks.append(task)
            except Exception as e:
                logger.error(f"Uncaught Thread Exception in {self.name} Dispatcher: {e}")

    def _on_task_done(self, task: IOTask, result, exception):
        # Receives (result, exception) from worker, then schedules task.on_complete
        caller_fut = self.caller_futures.pop(task, None)
        if caller_fut:
            if exception:
                caller_fut.set_exception(exception)
            else:
                caller_fut.set_result(result)

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
        self.dispatcher.shutdown(wait)
        self.executor.shutdown(wait)

    def join(self, timeout=None):
        # Block until all queued tasks processed (or until timeout)
        pass

    def clear_pending(self):
        # Remove all tasks still in queue
        while True:
            try:
                task = self.queue.get_nowait()
            except queue.Empty:
                break
            else:
                # Balance out get_nowait with a task_done
                self.queue.task_done()

        self.cleared_queue = True

    def clear_protocol_pending(self):
        while True:
            try:
                task = self.queue.get_nowait()
            except queue.Empty:
                break
            else:
                # Balance out get_nowait with a task_done
                self.queue.task_done()
        
        self.cleared_protocol_queue = True
    
    def is_busy(self):
        # Returns true if tasks queued or running
        return not (self.queue.empty() and self.running_task is None)

    

