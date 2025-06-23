
from kivy.clock import Clock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue

class IOTask:
        def __init__(self, action, args=(), kwargs={}, callback=None, cb_args=(), cb_kwargs={}, pass_result=False):
            self.action = action
            self.args = args
            self.kwargs = kwargs
            self.callback = callback
            self.cb_args = cb_args
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



class SequentialIOExecutor:
    def __init__(self, max_workers: int=1):
        self.queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.dispatcher = ThreadPoolExecutor(max_workers=1)
        self.running_task = None
        self.global_callback = None


    def start(self):
        # Start internal dispatcher
        self.dispatcher.submit(self._dispatch_loop)

    def enqueue(self, task: IOTask):
        # Push IO work item into queue
        self.queue.put(task)

    def _dispatch_loop(self):
        # Pulls from queue, submits to worker pool, wires up callbacks
        while True:
            task = self.queue.get(block=True)
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

    

