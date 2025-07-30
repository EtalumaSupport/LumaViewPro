import queue
import image_utils
from modules.video_writer import VideoWriter
import gc
import pathlib
from lvp_logger import logger


def write_capture(
                    protocol_execution_record:object=None,
                    separate_folder_per_channel:bool=False,
                    scope:object=None,
                    enable_image_saving=True,
                    is_video=False, 
                    video_as_frames=False, 
                    video_images: queue.Queue=None, 
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
                    captured_frames=1
                    ):
    
    print("===============WRITE CAPTURE ATTEMPT===============")
    print(f"is_video: {is_video}")
    print(f"video_as_frames: {video_as_frames}")
    print(f"video_images: {video_images}")
    print(f"save_folder: {save_folder}")
    print(f"use_color: {use_color}")
    print(f"name: {name}")
    print(f"calculated_fps: {calculated_fps}")
    print(f"output_format: {output_format}")

    print(f"step: {step}")
    print(f"captured_image: {captured_image}")
    print(f"step_index: {step_index}")
    print(f"scan_count: {scan_count}")
    print(f"capture_time: {capture_time}")
    print(f"duration_sec: {duration_sec}")
    print(f"captured_frames: {captured_frames}")
    print("===============WRITE CAPTURE ATTEMPT===============")

    if enable_image_saving == True:
        if is_video:
            if video_as_frames:
                frame_num = 0
                capture_result = save_folder
                if not save_folder.exists():
                    save_folder.mkdir(exist_ok=True, parents=True)

                while not video_images.empty():

                    image_pair = video_images.get_nowait()
                    frame_num += 1

                    image = image_pair[0]
                    ts = image_pair[1]

                    del image_pair

                    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                    image_w_timestamp = image_utils.add_timestamp(image=image, timestamp_str=ts_str)

                    del image
                    video_images.task_done()

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
                        logger.error(f"Protocol-Video] Failed to write frame {frame_num}: {e}")

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
                while not video_images.empty():
                    try:
                        image_pair = video_images.get_nowait()
                        video_writer.add_frame(image=image_pair[0], timestamp=image_pair[1])
                        del image_pair
                        video_images.task_done()
                    except Exception as e:
                        logger.error(f"Protocol-Video] FAILED TO WRITE FRAME: {e}")

                # Video images queue empty. Delete it and force garbage collection
                del video_images

                video_writer.finish()
                #video_writer.test_video(str(output_file_loc))
                del video_writer
                gc.collect()
                
                capture_result = output_file_loc
            
            logger.info("Protocol-Video] Video writing finished.")
            logger.info(f"Protocol-Video] Video saved at {capture_result}")
        
        else:
            if captured_image is False:
                return
            
            capture_result = scope.save_image(
                array=captured_image,
                save_folder=save_folder,
                file_root=None,
                append=name,
                color=use_color,
                tail_id_mode=None,
                output_format=output_format,
                true_color=step['Color'],
                x=step['X'],
                y=step['Y'],
                z=step['Z']
            )

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

    gc.collect()
    
    protocol_execution_record.add_step(
        capture_result_file_name=capture_result_filepath_name,
        step_name=step['Name'],
        step_index=step_index,
        scan_count=scan_count,
        timestamp=capture_time,
        frame_count=captured_frames if is_video else 1,
        duration_sec=duration_sec if is_video else 0.0
    )

    return
