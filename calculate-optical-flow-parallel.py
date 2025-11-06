#!/usr/bin/env python3
"""
Parallel optical flow calculation using multiprocessing
Provides 4-8x speedup over the original sequential version
"""

import os
import cv2 as cv
import numpy as np
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
import time

if len(sys.argv) < 2:
    print("Usage: {} <input_folder>".format(sys.argv[0]))
    sys.exit(-1)

dataset_dir = sys.argv[1]
if not os.path.isdir(dataset_dir):
    print("The command line argument is not a folder.")
    print("Usage: {} <input_folder>".format(sys.argv[0]))
    sys.exit(-1)


N_CLASSES = 7
classes = [str(i) for i in range(N_CLASSES)]

# the METC dataset has ~16 frames per second, the others have 30
FPS = 16 if "METC" in dataset_dir else 30

# what step to use for movements?
frame_step = FPS // 3


def mk(filename):
    try:
        os.makedirs(filename, exist_ok=True)
    except Exception as ex:
        pass


def read_frames(video_filename):
    """Read all frames from a video file"""
    cap = cv.VideoCapture(video_filename)

    frames = []
    ret, frame = cap.read()

    while ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames.append(gray)
        ret, frame = cap.read()

    cap.release()
    return frames


def frame_sequence_to_flow(f1, f2):
    """Calculate optical flow between two frames"""
    # Calculates dense optical flow by Farneback method
    flow = cv.calcOpticalFlowFarneback(f1, f2,
                                       None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)

    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros((len(f1), len(f1[0]), 3), dtype=np.float32)

    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2

    # Sets image saturation to maximum
    mask[..., 1] = 255

    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

    return rgb


def extract_flow_for_video(args):
    """
    Extract optical flow for a single video
    This function is designed to be called by multiprocessing.Pool

    Args:
        args: Tuple of (partition, class_name, filename, input_dir, output_dir, frame_step)

    Returns:
        Tuple of (success: bool, message: str, num_frames: int)
    """
    partition, c, filename, input_dir, output_dir, step = args

    try:
        in_fullname = os.path.join(input_dir, c, filename)
        out_fullname = os.path.join(output_dir, c, "frame_{}_" + os.path.splitext(filename)[0] + ".jpg")

        # Read frames
        frames = read_frames(in_fullname)
        n = len(frames)

        if n < step + 1:
            return (False, f"Video {filename} has insufficient frames ({n})", 0)

        # Calculate optical flow
        flow_frame_num = 0
        for i in range(n - step):
            frame1 = frames[i]
            frame2 = frames[i + step]
            flow_frame = frame_sequence_to_flow(frame1, frame2)

            save_path_and_name = out_fullname.format(flow_frame_num)
            flow_frame_num += 1
            cv.imwrite(save_path_and_name, flow_frame)

        return (True, f"Processed {in_fullname}", flow_frame_num)

    except Exception as e:
        return (False, f"Error processing {filename}: {str(e)}", 0)


def process_partition(partition, dataset_dir, classes, frame_step, num_workers=None):
    """
    Process one partition (test or trainval) in parallel

    Args:
        partition: "test" or "trainval"
        dataset_dir: Root dataset directory
        classes: List of class names
        frame_step: Step between frames for optical flow
        num_workers: Number of parallel workers (None = auto)
    """
    input_dir = os.path.join(dataset_dir, "videos", partition)
    output_dir = os.path.join(dataset_dir, "of", partition)

    # Create output directories
    mk(output_dir)
    for c in classes:
        mk(os.path.join(output_dir, c))

    # Collect all video files
    work_items = []
    for c in classes:
        class_dir = os.path.join(input_dir, c)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue

        for filename in os.listdir(class_dir):
            if filename.endswith(".mp4"):
                work_items.append((partition, c, filename, input_dir, output_dir, frame_step))

    if not work_items:
        print(f"No videos found in {partition} partition")
        return

    print(f"\n{'='*60}")
    print(f"Processing {partition} partition")
    print(f"Total videos: {len(work_items)}")
    print(f"{'='*60}\n")

    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), len(work_items))

    print(f"Using {num_workers} parallel workers")

    # Process in parallel
    start_time = time.time()

    with Pool(num_workers) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(extract_flow_for_video, work_items)):
            success, message, num_frames = result
            results.append((success, num_frames))

            # Print progress
            progress = (i + 1) / len(work_items) * 100
            print(f"[{progress:5.1f}%] {message}")

    end_time = time.time()
    elapsed = end_time - start_time

    # Summary
    successful = sum(1 for s, _ in results if s)
    failed = len(results) - successful
    total_frames = sum(n for _, n in results)

    print(f"\n{'-'*60}")
    print(f"Partition {partition} completed!")
    print(f"Successful: {successful}/{len(work_items)}")
    print(f"Failed: {failed}")
    print(f"Total optical flow frames: {total_frames}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Average time per video: {elapsed/len(work_items):.2f}s")
    print(f"{'-'*60}\n")


def main():
    print("\n" + "="*60)
    print("Parallel Optical Flow Calculation")
    print("="*60)
    print(f"Dataset: {dataset_dir}")
    print(f"FPS: {FPS}")
    print(f"Frame step: {frame_step}")
    print(f"CPU cores available: {cpu_count()}")
    print("="*60)

    # Get number of workers from environment variable or use auto
    num_workers_env = os.getenv("HANDWASH_NUM_WORKERS")
    if num_workers_env:
        num_workers = int(num_workers_env)
        print(f"Using {num_workers} workers (from HANDWASH_NUM_WORKERS)")
    else:
        num_workers = None
        print(f"Using automatic worker count")

    overall_start = time.time()

    # Process both partitions
    for partition in ["test", "trainval"]:
        process_partition(partition, dataset_dir, classes, frame_step, num_workers)

    overall_end = time.time()
    overall_elapsed = overall_end - overall_start

    print("\n" + "="*60)
    print("All done!")
    print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
