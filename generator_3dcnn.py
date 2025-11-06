"""
3D CNN data generator for video classification
Loads video clips as 3D tensors (time, height, width, channels)
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


def cnn3d_dataset_from_directory(video_directory,
                                  labels='inferred',
                                  label_mode='int',
                                  class_names=None,
                                  color_mode='rgb',
                                  num_frames=16,
                                  frame_step=2,
                                  batch_size=8,  # Smaller batch for 3D CNN
                                  image_size=(112, 112),  # Smaller images for 3D CNN
                                  shuffle=True,
                                  seed=None,
                                  validation_split=None,
                                  subset=None,
                                  interpolation='bilinear',
                                  follow_links=False,
                                  crop_to_aspect_ratio=False,
                                  **kwargs):
    """
    Generates a `tf.data.Dataset` from video frames for 3D CNN

    The key difference from TimeDistributed is that this loads consecutive frames
    as a 3D volume for 3D convolution operations.

    Args:
        video_directory: Directory containing frame images organized by class
        labels: Either "inferred", None, or list of labels
        label_mode: 'int', 'categorical', 'binary', or None
        class_names: List of class names
        color_mode: 'rgb', 'rgba', or 'grayscale'
        num_frames: Number of frames to load per video clip
        frame_step: Step between frames (frame rate subsampling)
        batch_size: Batch size (typically smaller for 3D CNN)
        image_size: Size to resize frames to (typically smaller, e.g., 112x112)
        shuffle: Whether to shuffle data
        seed: Random seed
        validation_split: Fraction for validation
        subset: "training" or "validation"
        interpolation: Interpolation method
        follow_links: Follow symlinks
        crop_to_aspect_ratio: Crop to maintain aspect ratio
        **kwargs: Additional arguments

    Returns:
        A `tf.data.Dataset` yielding (video_clip, label) tuples where
        video_clip has shape (batch_size, num_frames, height, width, channels)
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

    # Filter to only include frames that have enough consecutive frames
    filtered_image_paths = []
    filtered_labels = []
    image_path_set = set(image_paths)

    for path, label in zip(image_paths, labels):
        dirname = os.path.dirname(path)
        filename = os.path.basename(path)
        fields = filename.split("_")

        if len(fields) > 2:
            frame_num = int(fields[1])
            # Only use frames that can form complete clips
            chunk_size = num_frames * frame_step
            frame_pos_in_chunk = frame_num % chunk_size

            if frame_pos_in_chunk < frame_step:
                # Check if we have all required frames
                all_ok = True
                for i in range(num_frames):
                    fields[1] = str(frame_num + i * frame_step)
                    filename_check = "_".join(fields)
                    fullname = os.path.join(dirname, filename_check)
                    if fullname not in image_path_set:
                        all_ok = False
                        break

                if all_ok:
                    filtered_image_paths.append(path)
                    filtered_labels.append(label)

    image_paths = filtered_image_paths
    labels = filtered_labels

    # Apply train/val split
    image_paths, labels = dataset_utils.get_training_or_validation_split(
        image_paths, labels, validation_split, subset)

    if not image_paths:
        raise ValueError('No images found.')

    num_classes = len(class_names)

    # Create dataset
    img_dataset = paths_to_3d_dataset(
        image_paths,
        image_size,
        num_frames,
        frame_step,
        num_channels,
        num_classes,
        interpolation,
        crop_to_aspect_ratio)

    if label_mode:
        label_dataset = dataset_utils.labels_to_dataset(labels, label_mode, num_classes)
        dataset = dataset_ops.Dataset.zip((img_dataset, label_dataset))
    else:
        dataset = img_dataset

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)

    dataset = dataset.batch(batch_size)
    dataset.class_names = class_names
    dataset.file_paths = image_paths

    return dataset


def paths_to_3d_dataset(image_paths,
                        image_size,
                        num_frames,
                        frame_step,
                        num_channels,
                        num_classes,
                        interpolation,
                        crop_to_aspect_ratio=False):
    """Constructs a dataset of 3D video clips"""
    # Build list of frame sequences
    image_paths_multi = []
    for path in image_paths:
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        fields = basename.split("_")
        frame_num = int(fields[1])

        paths = []
        for i in range(num_frames):
            fields[1] = str(frame_num + i * frame_step)
            filename = "_".join(fields)
            fullname = os.path.join(dirname, filename)
            paths.append(fullname)

        image_paths_multi.append(paths)

    path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths_multi)
    args = (image_size, num_channels, interpolation, crop_to_aspect_ratio)
    img_ds = path_ds.map(
        lambda x: load_video_clip(x, *args),
        num_parallel_calls=tf.data.AUTOTUNE)

    return img_ds


def load_video_clip(paths, image_size, num_channels, interpolation,
                    crop_to_aspect_ratio=False):
    """Load a video clip as a 3D tensor"""
    imgs = []
    num_frames = len(paths)

    for i in range(num_frames):
        img = io_ops.read_file(paths[i])
        img = image_ops.decode_image(
            img, channels=num_channels, expand_animations=False)

        if crop_to_aspect_ratio:
            img = keras_image_ops.smart_resize(img, image_size,
                                              interpolation=interpolation)
        else:
            img = image_ops.resize_images_v2(img, image_size, method=interpolation)

        img.set_shape((image_size[0], image_size[1], num_channels))
        imgs.append(img)

    # Stack into 3D tensor (frames, height, width, channels)
    return tf.stack(imgs)
