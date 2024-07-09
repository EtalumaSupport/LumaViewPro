
import multiprocessing as mp


def get_usable_cpu_count() -> int:
    CPU_USAGE_RATIO = 0.75

    min_cpu_count = 1
    total_cpus = mp.cpu_count()
    target_cpu_count = round(total_cpus * CPU_USAGE_RATIO)
    
    if total_cpus == target_cpu_count:
        target_cpu_count -= 1

    target_cpu_count = max(target_cpu_count, min_cpu_count)
    return target_cpu_count
