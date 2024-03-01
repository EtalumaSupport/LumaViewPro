import cProfile
import datetime
import pathlib
import pstats


class ProfilingHelper:

    def __init__(
        self,
        save_path: pathlib.Path | None = None
    ):
        if save_path is None:
            self._profile_artifact_path = pathlib.Path(f'./logs/profile/{self._get_timestamp()}')
        else:
            self._profile_artifact_path = pathlib.Path(save_path)

        self._profile = None
        self._profile_artifact_path.mkdir(exist_ok=True, parents=True)


    def _get_timestamp(self):
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    

    def enable(self):
        self._profile = cProfile.Profile()
        self._profile.enable()


    def _disable_and_dump(self):
        ts = self._get_timestamp()
        self._profile.disable()
        self._profile.dump_stats(self._profile_artifact_path / f'LumaViewProApp_{ts}.profile')

        with open(self._profile_artifact_path / f'LumaViewProApp_{ts}.stats', 'w') as stream:
            stats = pstats.Stats(str(self._profile_artifact_path / f'LumaViewProApp_{ts}.profile'), stream=stream)
            stats.sort_stats('cumulative').dump_stats(str(self._profile_artifact_path / f'LumaViewProApp_{ts}_bin.stats'))
            # stats.print_stats()
            stats.sort_stats('cumulative').print_stats(30)
        

    def stop(self):
        if self._profile is not None:
            self._disable_and_dump()

    
    def restart(self, *args):
        self.stop()
        self.enable()
