import os
import glob
import argparse
import numpy as np

LEFT_WRIST = 15
RIGHT_WRIST = 16

def joint_motion_stats(poses, joint_idx):
    y = poses[:, joint_idx, 1]
    rom = float(y.max() - y.min())
    if len(y) > 1:
        vel = np.abs(np.diff(y))
        motion_energy = float(vel.mean())
    else:
        motion_energy = 0.0
    return rom, motion_energy

def simple_pose_feedback(poses):
    lw_rom, lw_me = joint_motion_stats(poses, LEFT_WRIST)
    rw_rom, rw_me = joint_motion_stats(poses, RIGHT_WRIST)

    rom = max(lw_rom, rw_rom)
    me  = max(lw_me,  rw_me)

    if rom < 0.05:
        return "Very little arm motion detected – this may not be an arm-focused exercise."
    elif me > 0.01:
        return "Arms are moving strongly and consistently throughout the clip."
    else:
        return "Moderate arm motion – movement looks relatively controlled."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_dir", type=str, required=True,
                        help="Folder containing *_pose.npy files")
    args = parser.parse_args()

    npy_paths = sorted(glob.glob(os.path.join(args.pose_dir, "*_pose.npy")))
    print(f"Found {len(npy_paths)} pose files in {args.pose_dir}")

    for npy_path in npy_paths:
        poses = np.load(npy_path)  # [T, 33, 4]
        if poses.shape[0] == 0:
            msg = "No pose detected for this clip."
        else:
            msg = simple_pose_feedback(poses)

        print(os.path.basename(npy_path).replace("_pose.npy", ".mp4"), "→", msg)

if __name__ == "__main__":
    main()
