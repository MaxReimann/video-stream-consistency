import os
import subprocess
import cv2

# THE  CORRECT ENCODER AND PIX-FLT ARE NOT SELECTED IN THE FFMPEG IMPLEMENTATION MAKE SURE TO DO SO BEFIRE USING
# def frames_to_video(frames_dir, output_video_dir, fps=30):
#     if not os.path.exists(frames_dir):
#         print(f"Error: Directory {frames_dir} does not exist.")
#         return
    
#     frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
#     if not frame_files:
#         print(f"Error: No .jpg files found in directory {frames_dir}.")
#         return

#     ffmpeg_command = [
#         "ffmpeg",
#         "-framerate", str(fps),
#         "-i", os.path.join(frames_dir, "%06d.jpg"),  
#         "-c:v", "libx264", 
#         "-pix_fmt", "yuv420p", 
#         output_video_dir
#     ]
#     try:
#         subprocess.run(ffmpeg_command, check=True)
#         print(f"Video saved to {output_video_dir}")
#     except subprocess.CalledProcessError as e:
#         print(f"Error: FFmpeg failed to create the video. {e}")


def frames_to_video_cv2(frames_dir, output_video_dir, output_video_name:str, fps:int = 30, output_video_codec:str = 'FFV1', output_video_container:str = 'mkv'):
    if not os.path.exists(frames_dir):
        print(f"Error: Directory {frames_dir} does not exist.")
        return

    if output_video_codec in ['FFV1', 'MJPG']:
        assert output_video_container in ['avi', 'mkv'], "Ensure that correct video container is used for the output_video_codec you are using"
    assert output_video_container in ['mkv', 'avi', 'mp4', 'mov', 'webm', 'flv', 'wmv'], 'Please use a suitable video containere'
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png")])
    if not frame_files:
        print(f"Error: No .jpg or .png files found in directory {frames_dir}.")
        return
    
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"Error: Could not read the first frame: {first_frame_path}")
        return
    height, width, _ = first_frame.shape
    frame_size = (width, height)
    
    fourcc = cv2.VideoWriter_fourcc(*f"{output_video_codec}")
    video_writer = cv2.VideoWriter(os.path.join(output_video_dir, f"{output_video_name}.{output_video_container}"), fourcc, fps, frame_size)
    
    for filename in frame_files:
        frame_path = os.path.join(frames_dir, filename)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}, skipping.")
            continue
        if (frame.shape[1], frame.shape[0]) != frame_size:
            print(f"Warning: Frame {frame_path} has different dimensions. Resizing.")
            frame = cv2.resize(frame, frame_size)
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Video saved to {output_video_dir}")


if __name__ == "__main__":
    frames_to_video_cv2(
        frames_dir = '/home/adity/VIDEO_SAVING_EXPERIMENTS/orignal_cv2_frames_png',
        output_video_dir = '/home/adity/VIDEO_SAVING_EXPERIMENTS',
        output_video_container = 'mkv',
        fps = 25,
        output_video_codec = 'MJPG',
        output_video_name = 'orignal_cv2_frames_png_MJPG_vid'
    )