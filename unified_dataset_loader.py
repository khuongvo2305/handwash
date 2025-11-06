"""
Unified dataset loader for all three datasets (Kaggle, PSKUS, METC)
Replaces the redundant dataset loading code across 9 different scripts
"""

import os
import numpy as np
import tensorflow as tf
from dataset_utilities import get_weights_dict
from generator_timedistributed import timedistributed_dataset_from_directory
from generator_rgb_with_of import merged_dataset_from_directories


class UnifiedDatasetLoader:
    """
    Unified interface for loading datasets for different model types
    """

    def __init__(self, config):
        """
        Initialize the loader with configuration

        Args:
            config: Dictionary containing dataset and training configuration
        """
        self.config = config
        self.training_config = config['training']

        # Extract common parameters
        self.img_size = (self.training_config['img_height'],
                        self.training_config['img_width'])
        self.batch_size = self.training_config['batch_size']
        self.n_classes = self.training_config['n_classes']
        self.validation_split = self.training_config['validation_split']
        self.seed = self.training_config['seed']

    def load_frame_dataset(self, dataset_name):
        """
        Load single-frame dataset for CNN models

        Args:
            dataset_name: Name of dataset ('kaggle', 'pskus', 'metc')

        Returns:
            Tuple of (train_ds, val_ds, test_ds, weights_dict)
        """
        dataset_config = self.config['datasets'][dataset_name]
        data_dir = dataset_config['paths']['frames_train']
        test_data_dir = dataset_config['paths']['frames_test']

        # Load training dataset
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=self.validation_split,
            subset="training",
            seed=self.seed,
            image_size=self.img_size,
            label_mode='categorical',
            crop_to_aspect_ratio=False,
            batch_size=self.batch_size
        )

        # Load validation dataset
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=self.validation_split,
            subset="validation",
            seed=self.seed,
            image_size=self.img_size,
            label_mode='categorical',
            crop_to_aspect_ratio=False,
            batch_size=self.batch_size
        )

        # Load test dataset
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            test_data_dir,
            seed=self.seed,
            image_size=self.img_size,
            label_mode='categorical',
            crop_to_aspect_ratio=False,
            batch_size=self.batch_size
        )

        # Calculate class weights
        weights_dict = get_weights_dict(data_dir, train_ds.class_names)

        # Apply optimizations
        train_ds, val_ds, test_ds = self._optimize_datasets(train_ds, val_ds, test_ds)

        return train_ds, val_ds, test_ds, weights_dict

    def load_video_dataset(self, dataset_name):
        """
        Load multi-frame video dataset for TimeDistributed models

        Args:
            dataset_name: Name of dataset ('kaggle', 'pskus', 'metc')

        Returns:
            Tuple of (train_ds, val_ds, test_ds, weights_dict)
        """
        dataset_config = self.config['datasets'][dataset_name]
        data_dir = dataset_config['paths']['frames_train']
        test_data_dir = dataset_config['paths']['frames_test']
        fps = dataset_config['fps']

        num_frames = self.training_config['num_frames']
        frame_step = fps // num_frames

        class_names = [str(i) for i in range(self.n_classes)]

        # Load training dataset
        train_ds = timedistributed_dataset_from_directory(
            data_dir,
            num_frames=num_frames,
            frame_step=frame_step,
            validation_split=self.validation_split,
            subset="training",
            seed=self.seed,
            image_size=self.img_size,
            shuffle=True,
            label_mode='categorical',
            batch_size=self.batch_size
        )

        # Load validation dataset
        val_ds = timedistributed_dataset_from_directory(
            data_dir,
            num_frames=num_frames,
            frame_step=frame_step,
            validation_split=self.validation_split,
            subset="validation",
            seed=self.seed,
            image_size=self.img_size,
            shuffle=True,
            label_mode='categorical',
            batch_size=self.batch_size
        )

        # Load test dataset
        test_ds = timedistributed_dataset_from_directory(
            test_data_dir,
            num_frames=num_frames,
            frame_step=frame_step,
            seed=self.seed,
            image_size=self.img_size,
            shuffle=False,
            label_mode='categorical',
            batch_size=self.batch_size
        )

        # Calculate class weights
        weights_dict = get_weights_dict(data_dir, class_names)

        return train_ds, val_ds, test_ds, weights_dict

    def load_merged_dataset(self, dataset_name):
        """
        Load RGB + Optical Flow dataset for two-stream models

        Args:
            dataset_name: Name of dataset ('kaggle', 'pskus', 'metc')

        Returns:
            Tuple of (train_ds, val_ds, test_ds, weights_dict)
        """
        dataset_config = self.config['datasets'][dataset_name]

        rgb_dir = dataset_config['paths']['frames_train']
        of_dir = dataset_config['paths']['of_train']
        test_rgb_dir = dataset_config['paths']['frames_test']
        test_of_dir = dataset_config['paths']['of_test']

        class_names = [str(i) for i in range(self.n_classes)]

        # Load training dataset
        train_ds = merged_dataset_from_directories(
            rgb_dir,
            of_dir,
            validation_split=self.validation_split,
            subset="training",
            seed=self.seed,
            image_size=self.img_size,
            shuffle=True,
            label_mode='categorical',
            crop_to_aspect_ratio=False,
            batch_size=self.batch_size
        )

        # Load validation dataset
        val_ds = merged_dataset_from_directories(
            rgb_dir,
            of_dir,
            validation_split=self.validation_split,
            subset="validation",
            seed=self.seed,
            image_size=self.img_size,
            shuffle=True,
            label_mode='categorical',
            crop_to_aspect_ratio=False,
            batch_size=self.batch_size
        )

        # Load test dataset
        test_ds = merged_dataset_from_directories(
            test_rgb_dir,
            test_of_dir,
            seed=self.seed,
            image_size=self.img_size,
            shuffle=False,
            label_mode='categorical',
            crop_to_aspect_ratio=False,
            batch_size=self.batch_size
        )

        # Calculate class weights
        weights_dict = get_weights_dict(rgb_dir, class_names)

        # Apply optimizations
        train_ds, val_ds, test_ds = self._optimize_datasets(train_ds, val_ds, test_ds)

        return train_ds, val_ds, test_ds, weights_dict

    def _optimize_datasets(self, train_ds, val_ds, test_ds):
        """
        Apply performance optimizations to datasets

        Args:
            train_ds, val_ds, test_ds: TensorFlow datasets

        Returns:
            Optimized datasets
        """
        AUTOTUNE = tf.data.AUTOTUNE

        # Apply caching if enabled
        if self.training_config.get('cache_dataset', True):
            train_ds = train_ds.cache()
            val_ds = val_ds.cache()
            test_ds = test_ds.cache()

        # Apply prefetching
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

        return train_ds, val_ds, test_ds


def load_dataset(config, dataset_name, model_type):
    """
    Convenience function to load datasets

    Args:
        config: Configuration dictionary
        dataset_name: Name of dataset ('kaggle', 'pskus', 'metc')
        model_type: Type of model ('frames', 'videos', 'merged')

    Returns:
        Tuple of (train_ds, val_ds, test_ds, weights_dict)
    """
    loader = UnifiedDatasetLoader(config)

    if model_type == 'frames':
        return loader.load_frame_dataset(dataset_name)
    elif model_type == 'videos':
        return loader.load_video_dataset(dataset_name)
    elif model_type == 'merged':
        return loader.load_merged_dataset(dataset_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Must be 'frames', 'videos', or 'merged'")
