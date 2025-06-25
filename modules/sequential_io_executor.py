
from kivy.clock import Clock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
from collections.abc import Sequence


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
    def __init__(self, max_workers: int=1):
        self.queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.dispatcher = ThreadPoolExecutor(max_workers=1)
        self.running_task = None
        self.global_callback = None
        self.pending_shutdown = False


    def start(self):
        # Start internal dispatcher
        self.dispatcher.submit(self._dispatch_loop)

    def put(self, task: IOTask):
        # Push IO work item into queue
        self.queue.put(task)

    def _dispatch_loop(self):
        # Pulls from queue, submits to worker pool, wires up callbacks
        while True:
            try:
                task = self.queue.get(block=True, timeout=0.2)
            except queue.Empty:
                if self.pending_shutdown:
                    return
                continue
            future = self.executor.submit(task.run)
            future.add_done_callback(lambda fut, t=task: self._on_task_done(t, *fut.result()))
            self.running_task = task
            self.queue.task_done()

    def _on_task_done(self, task: IOTask, result, exception):
        # Receives (result, exception) from worker, then schedules task.on_complete
        task.on_complete(result, exception)
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
        self.queue.queue.clear()
    
    def is_busy(self):
        # Returns true if tasks queued or running
        return not (self.queue.empty() and self.running_task is None)

    

