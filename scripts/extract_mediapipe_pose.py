import os
import glob
import argparse
import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

def extract_pose_sequence(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames = []
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        if max_frames is not None and frame_count > max_frames:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks is None:
            landmark_array = np.zeros((33, 4), dtype=np.float32)
        else:
            lm = results.pose_landmarks.landmark
            landmark_array = np.array(
                [[p.x, p.y, p.z, p.visibility] for p in lm],
                dtype=np.float32,
            )

        frames.append(landmark_array)

    cap.release()
    pose.close()

    if len(frames) == 0:
        return np.zeros((0, 33, 4), dtype=np.float32)

    return np.stack(frames, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Folder with input videos (.mp4/.mov/.avi)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Folder to save pose .npy files")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Optional cap on number of videos to process")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    video_paths = sorted(
        glob.glob(os.path.join(args.video_dir, "*.mp4"))
        + glob.glob(os.path.join(args.video_dir, "*.mov"))
        + glob.glob(os.path.join(args.video_dir, "*.avi"))
    )

    if args.max_videos is not None:
        video_paths = video_paths[: args.max_videos]

    print(f"Found {len(video_paths)} videos in {args.video_dir}")

    for vid_path in video_paths:
        base = os.path.splitext(os.path.basename(vid_path))[0]
        out_path = os.path.join(args.out_dir, f"{base}_pose.npy")

        if os.path.exists(out_path):
            print(f"[SKIP] {base} (already exists)")
            continue

        print(f"[PROCESS] {base}")
        poses = extract_pose_sequence(vid_path)
        print("  -> shape:", poses.shape)
        np.save(out_path, poses)

    print("Done.")

if __name__ == "__main__":
    main()
