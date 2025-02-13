import os
import subprocess
import time
import shutil
from tqdm import tqdm
import tempfile
import glob

from frametovid import frames_to_video_cv2
from vidtoframe import save_video_frames

def run_flow_consistency(gt_frame_dir:str, processed_frame_dir:str, generated_frame_dir:str, downscalingfactor:int = None):
    if not os.path.exists(generated_frame_dir):
        os.mkdirs(generated_frame_dir)
    
    if downscalingfactor != None:
        commands = [
            f"unset FLOWDOWNSCALE && export FLOWDOWNSCALE={downscalingfactor} && LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH && cd build && ./FlowVideoConsistency -c pwcnet-light {gt_frame_dir} {processed_frame_dir} {generated_frame_dir}"
        ]
    else:
        commands = [
            f"unset FLOWDOWNSCALE && LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH && cd build && ./FlowVideoConsistency -c pwcnet-light {gt_frame_dir} {processed_frame_dir} {generated_frame_dir}"
        ]
    
    result = subprocess.run(commands, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

def multi_inference(gt_frame_dir:str, initial_processed_dir:str, final_generated_dir:str, downscalingfactor:int, num_inferences:int, fps:float, output_video_codec:str, output_video_name:str):
    with tempfile.TemporaryDirectory() as temp_workspace:
        working_processed_dir = os.path.join(temp_workspace, 'processed_0')
        shutil.copytree(initial_processed_dir, working_processed_dir)

        for i in tqdm(range(num_inferences), desc="inferences"):
            this_generated_dir = os.path.join(temp_workspace, f'generated_{i+1}')
            os.makedirs(this_generated_dir, exist_ok=True)

            run_flow_consistency(gt_frame_dir, working_processed_dir, this_generated_dir, downscalingfactor)

            if i <num_inferences-1:
                new_working_processed_dir = os.path.join(temp_workspace, f"processed_{i+1}")
                os.makedirs(new_working_processed_dir, exist_ok=True)

                frames = sorted(
                    glob.glob(os.path.join(this_generated_dir, '*.jpg')) +
                    glob.glob(os.path.join(this_generated_dir, '*.png'))
                )
                num_frames = len(frames)
                if num_frames == 0:
                    raise Exception(f'no frames were actually found in generation round {i+1}')

                image_container = frames[0].split('.')[-1]

                for idx, frame_address in enumerate(frames, start=1):
                    name = f'{idx:06d}.{image_container}'
                    destination = os.path.join(new_working_processed_dir, name)
                    shutil.copy2(frame_address, destination)
                
                new_frames = sorted(
                    glob.glob(os.path.join(this_generated_dir, '*.jpg')) +
                    glob.glob(os.path.join(this_generated_dir, '*.png'))
                )
                if len(new_frames) != num_frames:
                    raise Exception("Mismatch in number of frames after renaming after round ", i+1)
            
                working_processed_dir = new_working_processed_dir
            
            else:

                if not os.path.exists(final_generated_dir):
                    os.makedirs(final_generated_dir, exist_ok=True)
                
                frames_to_video_cv2(
                    frames_dir = this_generated_dir,
                    output_video_dir = final_generated_dir,
                    output_video_name = output_video_name,
                    fps = fps,
                    output_video_codec = output_video_codec,
                    output_video_container = output_video_container,
                )

                for item in os.listdir(this_generated_dir):
                    source = os.path.join(this_generated_dir, item)
                    destination = os.path.join(final_generated_dir, item)
                    if os.path.isdir(source):
                        shutil.copytree(source, destination, dirs_exist_ok=True)
                    else:
                        shutil.copy2(source, destination)


if __name__ == "__main__":
    gt_frame_dir = '/home/adity/data/VSC_data/input/6_GT'
    initial_processed_dir = '/home/adity/data/VSC_data/processed/6_processed'
    final_generated_dir = '/home/adity/data/VSC_data/generation_v2/6_light_twice_generated_config_def_direct_internally_not_integrated_pngs'
    downscalingfactor = None
    num_inferences = 2
    fps = 25
    output_video_codec = 'MJPG'
    output_video_name = 'final_generation'
    output_video_container = 'mkv'

    multi_inference(
        gt_frame_dir=gt_frame_dir,
        initial_processed_dir=initial_processed_dir,
        final_generated_dir=final_generated_dir,
        downscalingfactor=downscalingfactor,
        num_inferences=num_inferences,
        fps=fps,
        output_video_codec=output_video_codec,
        output_video_name= output_video_name
    )