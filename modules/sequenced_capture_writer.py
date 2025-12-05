import queue
import image_utils
from modules.video_writer import VideoWriter
import gc
import pathlib
# from lvp_logger import logger
import os
import logging

# Prevent any potential Kivy imports in worker processes
os.environ['KIVY_NO_CONSOLELOG'] = '1'

def _noop(i: int):
    print(f"noop task received: {i!r}")
    return None


class FlushFileHandler(logging.FileHandler):
    """FileHandler that flushes after each emit."""
    def emit(self, record):
        super().emit(record)
        self.flush()

def setup_worker_logger(log_dir: str = ".") -> logging.Logger:
    """
    Create and return a logger that writes to
    ./worker_<PID>.log, flushing after every record.
    """
    pid = os.getpid()   
    name = f"worker_{pid}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Only add the handler once:
    if not any(isinstance(h, FlushFileHandler) for h in logger.handlers):
        log_path = os.path.join(log_dir, f"{name}.log")
        fh = FlushFileHandler(log_path, mode="w")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def worker_initializer(lvp_appdata):
    try:
        """Initialize worker process - called once when process starts."""
        import os
        import sys
        import builtins
        from datetime import datetime
    
        pid = os.getpid()
        # Open two files, line-buffered
        sys.stdout = open(f"{lvp_appdata}/logs/subprocess_workers.log", "a", buffering=1)
        sys.stderr = open(f"{lvp_appdata}/logs/subprocess_workers.error.log", "a", buffering=1)

        orig_print = builtins.print
        def prefixed_print(*args, **kwargs):
            # Always print to our new sys.stdout
            timestamp = datetime.now().isoformat(sep=" ", timespec="milliseconds")
            prefix = f"{timestamp} [PID {pid}]"
            orig_print(prefix, *args, **kwargs, file=sys.stdout, flush=True)

        builtins.print = prefixed_print

        # Set environment variables to prevent Kivy initialization
        os.environ['KIVY_NO_CONSOLELOG'] = '1'
        os.environ['KIVY_NO_ARGS'] = '1'
        os.environ['KIVY_NO_CONFIG'] = '1'
        os.environ['KIVY_LOGGER_LEVEL'] = 'critical'
        
        # Prevent pygame sound initialization
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        print(f"Worker process {pid} initialized")
        print("initializer complete")
        # Set up any worker-specific configuration here
        # This runs once per worker process at startup
        pass
    except Exception:
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        raise

def write_capture(
                    protocol_execution_record:object=None,
                    separate_folder_per_channel:bool=False,
                    save_image_func:callable=None,
                    save_image_func_kwargs:dict=None,
                    enable_image_saving=True,
                    is_video=False, 
                    video_as_frames=False, 
                    video_images: list=None, 
                    save_folder=None,
                    use_color=None,
                    name=None,
                    calculated_fps=None,
                    output_format=None,
                    step=None,
                    captured_image=None,
                    step_index=None,
                    scan_count=None,
                    capture_time=None,
                    duration_sec=0.0,
                    captured_frames=1,
                    thread=False
                    ):

    if thread:
        from lvp_logger import logger
        logger.info(f"Protocol-Writer] Begin write_capture for {name}")
    else:
        print(f"Begin write_capture for {name}")

    try:
        if enable_image_saving == True:
            if is_video:
                if video_as_frames:
                    frame_num = 0
                    capture_result = save_folder
                    if not save_folder.exists():
                        save_folder.mkdir(exist_ok=True, parents=True)

                    for image_pair in video_images:

                        frame_num += 1

                        image = image_pair[0]
                        ts = image_pair[1]

                        del image_pair

                        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                        image_w_timestamp = image_utils.add_timestamp(image=image, timestamp_str=ts_str)

                        frame_name = f"{name}_Frame_{frame_num:04}"

                        output_file_loc = save_folder / f"{frame_name}.tiff"

                        metadata = {
                            "datetime": ts.strftime("%Y:%m:%d %H:%M:%S"),
                            "timestamp": ts.strftime("%Y:%m:%d %H:%M:%S.%f"),
                            "frame_num": frame_num
                        }
                        
                        try:
                            image_utils.write_tiff(
                                data=image_w_timestamp,
                                metadata=metadata,
                                file_loc=output_file_loc,
                                video_frame=True,
                                ome=False,
                            )
                        except Exception as e:
                            print(f"Protocol-Video] Failed to write frame {frame_num}: {e}")

                        del image_w_timestamp
                        del image
                        del ts

                    # Queue is not empty, delete it and force garbage collection
                    del video_images
                    gc.collect()

                        

                else:
                    output_file_loc = save_folder / f"{name}.mp4v"
                    video_writer = VideoWriter(
                        output_file_loc=output_file_loc,
                        fps=calculated_fps,
                        include_timestamp_overlay=True
                    )
                    for image_pair in video_images:
                        try:
                            video_writer.add_frame(image=image_pair[0], timestamp=image_pair[1])
                            del image_pair
                        except Exception as e:
                            print(f"Protocol-Video] FAILED TO WRITE FRAME: {e}")

                    # Video images queue empty. Delete it and force garbage collection
                    del video_images

                    video_writer.finish()
                    #video_writer.test_video(str(output_file_loc))
                    del video_writer
                    gc.collect()
                    
                    capture_result = output_file_loc
                
                print("Protocol-Video] Video writing finished.")
                print(f"Protocol-Video] Video saved at {capture_result}")
            
            else:
                if captured_image is False:
                    return
                
                # save_image_func is a static method on the Lumascope class
                # save_image_func_kwargs is a dictionary of keyword arguments to pass to the save_image_func
                # save_image_func should be Lumascope.save_image_static(kwargs)
                try:
                    capture_result = save_image_func(
                        **save_image_func_kwargs
                    )
                    print(f"Image saved to {capture_result}")

                except Exception as e:
                    print(f"Error: Unable to save image: {e}")
                    raise Exception(f"Unable to save image: {e}")

                del captured_image
                gc.collect()

                
                
                # result = self._scope.save_live_image(
                #     save_folder=save_folder,
                #     file_root=None,
                #     append=name,
                #     color=use_color,
                #     tail_id_mode=None,
                #     force_to_8bit=not use_full_pixel_depth,
                #     output_format=output_format,
                #     true_color=step['Color'],
                #     earliest_image_ts=earliest_image_ts,
                #     timeout=datetime.timedelta(seconds=1.0),
                #     all_ones_check=True,
                #     sum_count=sum_count,
                #     sum_delay_s=step["Exposure"]/1000,
                #     sum_iteration_callback=sum_iteration_callback,
                #     turn_off_all_leds_after=True,
                # )
            if capture_result is None:
                capture_result_filepath_name = "unsaved"

            elif type(capture_result) == dict:
                capture_result_filepath_name = capture_result['metadata']['file_loc']

            elif separate_folder_per_channel:
                capture_result_filepath_name = pathlib.Path(step["Color"]) / capture_result.name

            else:
                capture_result_filepath_name = capture_result.name

        else:
            capture_result_filepath_name = "unsaved"

    except Exception as e:
        if thread:
            from lvp_logger import logger
            logger.error(f"Protocol-Writer] Error: Unable to save image: {e}")
        else:
            print(f"Error: Unable to save image: {e}")
        raise Exception(f"Unable to save image: {e}")

    gc.collect()

    if thread:
        logger.info(f"Protocol-Writer] Successful return of {capture_result_filepath_name}")
    else:
        print(f"Successful return of {capture_result_filepath_name}")

    if thread:
        from modules.sequenced_capture_executor import write_finished_callback
        write_finished_callback(
            protocol_execution_record=protocol_execution_record,
            step_name=name,
            step_index=step_index,
            scan_count=scan_count,
            timestamp=capture_time,
            frame_count=captured_frames,
            duration_sec=duration_sec,
            fut=None,
            result=capture_result_filepath_name,
            exception=None
        )
        return

    return capture_result_filepath_name