import tracemalloc
import gc
import os
import time
import threading
from datetime import datetime

class MemoryLeakProfiler:
    # 1. Configuration
    INTERVAL = 900  # Take a snapshot every 15 minutes (900s)
    TRACE_DEPTH = 25 # How many stack frames to record
    LOG_DIR = datetime.now().strftime("%Y%m%d_%H%M%S") + "_memory_logs"

    
    _first_snapshot = None

    @classmethod
    def start(cls, root_log_dir="."):
        cls.LOG_DIR = os.path.join(root_log_dir, cls.LOG_DIR)
        if not os.path.exists(cls.LOG_DIR):
            os.makedirs(cls.LOG_DIR)
            
        # Start tracing with deep stack history
        tracemalloc.start(cls.TRACE_DEPTH)
        cls._first_snapshot = tracemalloc.take_snapshot()
        
        # Run in a background thread so it doesn't lag the Kivy UI
        threading.Thread(target=cls._profiler_loop, daemon=True).start()
        print(f"[Profiler] Started. Logs saving to {cls.LOG_DIR}")

    @classmethod
    def _profiler_loop(cls):
        while True:
            time.sleep(cls.INTERVAL)
            cls.take_report()

    @classmethod
    def take_report(cls):
        # Force GC to ensure we only see 'true' leaks, not pending cleanup
        gc.collect()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_snapshot = tracemalloc.take_snapshot()

        # Binary Save: Preserves 100% of the data for offline analysis
        binary_path = os.path.join(cls.LOG_DIR, f"snapshot_{timestamp}.bin")
        current_snapshot.dump(binary_path)

        # Log 1: Current vs Start (Total accumulation)
        cls._save_comparison(cls._first_snapshot, current_snapshot, f"growth_since_start_{timestamp}.txt")
        
        # Log 2: Current total memory usage stats
        cls._save_lineno_stats(current_snapshot, f"total_usage_lineno_{timestamp}.txt")
        #cls._save_traceback_stats(current_snapshot, f"total_usage_traceback_{timestamp}.txt")
        
        # Log 3: Summary of peak memory
        current, peak = tracemalloc.get_traced_memory()
        with open(os.path.join(cls.LOG_DIR, "summary.log"), "a") as f:
            f.write(f"[{timestamp}] Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB\n")

    @classmethod
    def _save_comparison(cls, old_snap, new_snap, filename):
        # 'traceback' grouping is the "True Root" tool—it shows the full call path
        stats = new_snap.compare_to(old_snap, 'traceback')
        
        filepath = os.path.join(cls.LOG_DIR, filename)
        with open(filepath, "w") as f:
            f.write(f"Top 25 Memory Growth Sources (by Traceback):\n")
            for stat in stats[:25]:
                f.write(f"\n--- {stat.size_diff / 1024:.2f} KiB increase ---\n")
                for line in stat.traceback.format():
                    f.write(f"{line}\n")

    @classmethod
    def _save_lineno_stats(cls, snapshot, filename):
        stats = snapshot.statistics('lineno')
        filepath = os.path.join(cls.LOG_DIR, filename)
        with open(filepath, "w") as f:
            f.write("Top 25 Current Allocations (by Line Number):\n")
            for stat in stats[:25]:
                f.write(f"{stat}\n")

    @classmethod
    def _save_traceback_stats(cls, snapshot, filename):
        stats = snapshot.statistics('traceback')
        filepath = os.path.join(cls.LOG_DIR, filename)
        with open(filepath, "w") as f:
            f.write("Top 25 Current Allocations (by Traceback):\n")
            for stat in stats[:25]:
                f.write(f"\n--- {stat.size / 1024:.2f} KiB allocated ---\n")
                for line in stat.traceback.format():
                    f.write(f"{line}\n")
