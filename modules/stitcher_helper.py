
import multiprocessing as mp

if __name__ == "__main__":
    mp.freeze_support()


def run_stitcher_mp(num_workers: int, items):
    with mp.Pool(num_workers) as pool:
        result = pool.map(run_stitch_on_group, items)

    return result


def run_stitch_on_group(self, group):
    print('hi')
