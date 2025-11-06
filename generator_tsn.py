"""
Temporal Segment Network (TSN) data generator
Divides video into segments and samples one frame from each segment
Efficient for processing long videos
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras.layers.preprocessing import image_preprocessing
from tensorflow.python.keras.preprocessing import dataset_utils
from tensorflow.python.keras.preprocessing import image as keras_image_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops

ALLOWLIST_FORMATS = ('.bmp', '.gif', '.jpeg', '.jpg', '.png')


def tsn_dataset_from_directory(video_directory,
                                labels='inferred',
                                label_mode='int',
                                class_names=None,
                                color_mode='rgb',
                                num_segments=3,
                                frames_per_video=30,  # Approximate total frames
                                batch_size=32,
                                image_size=(224, 224),
                                shuffle=True,
                                seed=None,
                                validation_split=None,
                                subset=None,
                                interpolation='bilinear',
                                follow_links=False,
                                crop_to_aspect_ratio=False,
                                **kwargs):
    """
    Generates a `tf.data.Dataset` for Temporal Segment Networks

    TSN divides each video into N segments and samples 1 frame from each segment.
    This captures the temporal evolution of the video efficiently.

    Args:
        video_directory: Directory containing frame images organized by class
        labels: Either "inferred", None, or list of labels
        label_mode: 'int', 'categorical', 'binary', or None
        class_names: List of class names
        color_mode: 'rgb', 'rgba', or 'grayscale'
        num_segments: Number of segments to divide video into
        frames_per_video: Approximate number of frames per video (for sampling)
        batch_size: Batch size
        image_size: Size to resize frames to
        shuffle: Whether to shuffle data
        seed: Random seed
        validation_split: Fraction for validation
        subset: "training" or "validation"
        interpolation: Interpolation method
        follow_links: Follow symlinks
        crop_to_aspect_ratio: Crop to maintain aspect ratio
        **kwargs: Additional arguments

    Returns:
        A `tf.data.Dataset` yielding ([segment_0, segment_1, ..., segment_N], label) tuples
        where each segment has shape (batch_size, height, width, channels)
    """
    if 'smart_resize' in kwargs:
        crop_to_aspect_ratio = kwargs.pop('smart_resize')
    if kwargs:
        raise TypeError(f'Unknown keywords argument(s): {tuple(kwargs.keys())}')

    if labels not in ('inferred', None):
        if not isinstance(labels, (list, tuple)):
            raise ValueError(
                '`labels` argument should be a list/tuple of integer labels, of '
                'the same size as the number of image files in the target '
                'directory. If you wish to infer the labels from the subdirectory '
                'names in the target directory, pass `labels="inferred"`. '
                'If you wish to get a dataset that only contains images '
                '(no labels), pass `label_mode=None`.')
        if class_names:
            raise ValueError('You can only pass `class_names` if the labels are '
                           'inferred from the subdirectory names in the target '
                           'directory (`labels="inferred"`).')

    if label_mode not in {'int', 'categorical', 'binary', None}:
        raise ValueError(
            '`label_mode` argument must be one of "int", "categorical", "binary", '
            'or None. Received: %s' % (label_mode,))

    if labels is None or label_mode is None:
        labels = None
        label_mode = None

    if color_mode == 'rgb':
        num_channels = 3
    elif color_mode == 'rgba':
        num_channels = 4
    elif color_mode == 'grayscale':
        num_channels = 1
    else:
        raise ValueError(
            '`color_mode` must be one of {"rbg", "rgba", "grayscale"}. '
            'Received: %s' % (color_mode,))

    interpolation = image_preprocessing.get_interpolation(interpolation)
    dataset_utils.check_validation_split_arg(validation_split, subset, shuffle, seed)

    if seed is None:
        seed = np.random.randint(1e6)

    # Index all frame files
    image_paths, labels, class_names = dataset_utils.index_directory(
        video_directory,
        labels,
        formats=ALLOWLIST_FORMATS,
        class_names=class_names,
        shuffle=shuffle,
        seed=seed,
        follow_links=follow_links)

    if label_mode == 'binary' and len(class_names) != 2:
        raise ValueError(
            'When passing `label_mode="binary", there must exactly 2 classes. '
            'Found the following classes: %s' % (class_names,))

    # Group frames by video
    # Assuming naming: frame_0_videoname.jpg, frame_1_videoname.jpg, etc.
    video_frames = {}
    for path in image_paths:
        filename = os.path.basename(path)
        fields = filename.split("_")
        if len(fields) > 2:
            video_name = "_".join(fields[2:])  # Everything after frame number
            if video_name not in video_frames:
                video_frames[video_name] = []
            video_frames[video_name].append(path)

    # Sort frames within each video and sample segments
    video_paths_list = []
    video_labels_list = []

    for video_name, frame_paths in video_frames.items():
        # Sort by frame number
        frame_paths.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))

        # Only use videos with enough frames
        if len(frame_paths) >= num_segments:
            # Sample one frame from each segment
            segment_size = len(frame_paths) // num_segments
            sampled_frames = []

            for seg_idx in range(num_segments):
                start_idx = seg_idx * segment_size
                end_idx = start_idx + segment_size if seg_idx < num_segments - 1 else len(frame_paths)

                # Sample from middle of segment (can be randomized during training)
                mid_idx = (start_idx + end_idx) // 2
                sampled_frames.append(frame_paths[mid_idx])

            video_paths_list.append(sampled_frames)

            # Get label from first frame of video
            first_frame_idx = image_paths.index(frame_paths[0])
            video_labels_list.append(labels[first_frame_idx])

    # Apply train/val split
    video_paths_list, video_labels_list = dataset_utils.get_training_or_validation_split(
        video_paths_list, video_labels_list, validation_split, subset)

    if not video_paths_list:
        raise ValueError('No videos found.')

    num_classes = len(class_names)

    # Create dataset
    datasets = []
    for seg_idx in range(num_segments):
        segment_paths = [video_paths[seg_idx] for video_paths in video_paths_list]
        path_ds = dataset_ops.Dataset.from_tensor_slices(segment_paths)
        args = (image_size, num_channels, interpolation, crop_to_aspect_ratio)
        img_ds = path_ds.map(
            lambda x: load_image(x, *args),
            num_parallel_calls=tf.data.AUTOTUNE)
        datasets.append(img_ds)

    # Zip all segment datasets together
    img_dataset = dataset_ops.Dataset.zip(tuple(datasets))

    if label_mode:
        label_dataset = dataset_utils.labels_to_dataset(video_labels_list, label_mode, num_classes)
        dataset = dataset_ops.Dataset.zip((img_dataset, label_dataset))
    else:
        dataset = img_dataset

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)

    dataset = dataset.batch(batch_size)
    dataset.class_names = class_names
    dataset.file_paths = video_paths_list

    return dataset


def load_image(path, image_size, num_channels, interpolation,
               crop_to_aspect_ratio=False):
    """Load a single image"""
    img = io_ops.read_file(path)
    img = image_ops.decode_image(
        img, channels=num_channels, expand_animations=False)

    if crop_to_aspect_ratio:
        img = keras_image_ops.smart_resize(img, image_size,
                                          interpolation=interpolation)
    else:
        img = image_ops.resize_images_v2(img, image_size, method=interpolation)

    img.set_shape((image_size[0], image_size[1], num_channels))
    return img
