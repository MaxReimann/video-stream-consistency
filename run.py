import os
import subprocess
import time

from frametovid import frames_to_video
from vidtoframe import save_video_frames

def run_flow_consistency(gt_frame_dir:str, processed_frame_dir:str, generated_frame_dir:str, downscalingfactor:int = None):
    if not os.path.exists(generated_frame_dir):
        os.makedirs(generated_frame_dir)

    if downscalingfactor != None:
        commands = [
            f"unset FLOWDOWNSCALE && export FLOWDOWNSCALE={downscalingfactor} && LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH && cd build && ./FlowVideoConsistency -c pwcnet-light {gt_frame_dir} {processed_frame_dir} {generated_frame_dir}"
        ]
    else:
        commands = [
            f"unset FLOWDOWNSCALE && LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH && cd build && ./FlowVideoConsistency -c pwcnet-light {gt_frame_dir} {processed_frame_dir} {generated_frame_dir}"
        ]
    t1 = time.perf_counter()
    for command in commands:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    t2 = time.perf_counter()
    
    inference_time = t2 - t1
    print(f"INFERENCE TIME TAKEN = {inference_time:.4f} s")

if __name__ == "__main__":
    gt_frame_dir = '/home/adity/data/VSC_data/input/6_GT'
    processed_frame_dir = '/home/adity/data/VSC_data/processed/6_processed'
    generated_frame_dir = '/home/adity/data/VSC_data/generation/6_light_thrice_generated_downscale4_config_def'

    run_flow_consistency(gt_frame_dir, processed_frame_dir, generated_frame_dir, downscalingfactor=4)

    frames_to_video(
        frames_dir = generated_frame_dir,
        output_video_path = os.path.join(generated_frame_dir, generated_frame_dir.split('/')[-1] + '.mp4'),
        fps = 30
    )

    save_video_frames(
        video_path= os.path.join(generated_frame_dir, generated_frame_dir.split('/')[-1] + '.mp4'),
        destination_dir='home/adity/processed_once'
    )

    run_flow_consistency(gt_frame_dir, 'home/adity/processed_once', 'home/adity/generated_twice')