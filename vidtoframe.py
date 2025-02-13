import cv2
import os

def save_video_frames(video_path, destination_dir, image_container_type:str = 'jpg'):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    assert image_container_type in ['png', 'jpg'], "Only applicable vlaues for image_container are 'png' or 'jpg'"

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(destination_dir, f"{frame_count:06}.{image_container_type}")
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()
    print(f"Saved {frame_count} frames to {destination_dir}")


if __name__ == "__main__":
    save_video_frames(
        video_path = '/home/adity/data/input_videos/6_processed.mp4',
        destination_dir = '/home/adity/data/VSC_data/processed/6_processed_png',
        image_container_type = 'jpg'
    )

