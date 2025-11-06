#!/usr/bin/env python3
"""
Unified training script for hand washing movement classification

This single script replaces 9 redundant training scripts:
- kaggle-classify-frames.py, pskus-classify-frames.py, rsu-metc-classify-frames.py
- kaggle-classify-videos.py, pskus-classify-videos.py, rsu-metc-classify-videos.py
- kaggle-classify-merged-network.py, pskus-classify-merged-network.py, rsu-metc-classify-merged-network.py

Usage:
    python train.py --dataset kaggle --model frames
    python train.py --dataset pskus --model videos
    python train.py --dataset metc --model merged
    python train.py --dataset kaggle --model frames --config custom_config.yaml
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Import training modules
from classify_dataset import (
    evaluate,
    get_default_model,
    get_time_distributed_model,
    get_merged_model,
    get_3dcnn_model,
    get_tsn_model,
    get_i3d_model,
    get_slowfast_model
)
from unified_dataset_loader import load_dataset


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def apply_config_to_env(config):
    """
    Apply configuration to environment variables
    This ensures compatibility with existing classify_dataset.py code
    """
    training_config = config['training']

    # Set environment variables that classify_dataset.py reads
    os.environ['HANDWASH_NN'] = training_config.get('model_name', 'MobileNetV2')
    os.environ['HANDWASH_NUM_LAYERS'] = str(training_config.get('num_trainable_layers', 0))
    os.environ['HANDWASH_NUM_EPOCHS'] = str(training_config.get('num_epochs', 20))
    os.environ['HANDWASH_NUM_FRAMES'] = str(training_config.get('num_frames', 5))
    os.environ['HANDWASH_EXTRA_LAYERS'] = str(training_config.get('num_extra_layers', 0))


def get_model_builder(model_type, num_segments=3):
    """Get the appropriate model builder function"""
    if model_type == 'frames':
        return get_default_model
    elif model_type == 'videos':
        return get_time_distributed_model
    elif model_type == 'merged':
        return get_merged_model
    elif model_type == '3dcnn':
        return get_3dcnn_model
    elif model_type == 'tsn':
        return lambda: get_tsn_model(num_segments)
    elif model_type == 'i3d':
        return get_i3d_model
    elif model_type == 'slowfast':
        return get_slowfast_model
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Unified training script for hand washing classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single-frame CNN on Kaggle dataset
  python train.py --dataset kaggle --model frames

  # Train video model with GRU on PSKUS dataset
  python train.py --dataset pskus --model videos

  # Train two-stream RGB+OF model on METC dataset
  python train.py --dataset metc --model merged

  # Use custom configuration file
  python train.py --dataset kaggle --model frames --config my_config.yaml

  # Override epochs via environment variable
  HANDWASH_NUM_EPOCHS=50 python train.py --dataset kaggle --model frames
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['kaggle', 'pskus', 'metc'],
        help='Dataset to use for training'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['frames', 'videos', 'merged', '3dcnn', 'tsn', 'i3d', 'slowfast'],
        help='Model type: frames, videos, merged, 3dcnn, tsn, i3d (Inflated 3D), slowfast (SlowFast Networks)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/datasets.yaml',
        help='Path to configuration file (default: config/datasets.yaml)'
    )

    parser.add_argument(
        '--suffix',
        type=str,
        default='',
        help='Suffix to add to output files'
    )

    args = parser.parse_args()

    # Check if config file exists
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Check if dataset exists in config
    if args.dataset not in config['datasets']:
        print(f"Error: Dataset '{args.dataset}' not found in configuration")
        sys.exit(1)

    # Apply configuration to environment variables
    apply_config_to_env(config)

    # Override suffix if provided
    if args.suffix:
        os.environ['HANDWASH_SUFFIX'] = args.suffix

    # Print training information
    dataset_config = config['datasets'][args.dataset]
    print("\n" + "="*60)
    print(f"Training Configuration")
    print("="*60)
    print(f"Dataset: {dataset_config['name']} ({args.dataset})")
    print(f"Model: {args.model}")
    print(f"FPS: {dataset_config['fps']}")
    print(f"Base Model: {config['training']['model_name']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Image Size: {config['training']['img_height']}x{config['training']['img_width']}")

    if args.model in ['videos', '3dcnn', 'i3d', 'slowfast']:
        print(f"Num Frames: {config['training'].get('num_frames', 5)}")

    if args.model == 'tsn':
        print(f"Num Segments: {config['training'].get('num_segments', 3)}")

    if args.model == 'slowfast':
        print(f"Alpha (temporal ratio): {config['training'].get('slowfast_alpha', 4)}")
        print(f"Beta (channel ratio): {config['training'].get('slowfast_beta', 8)}")

    print("="*60 + "\n")

    # Load dataset
    print(f"Loading {args.model} dataset for {args.dataset}...")
    train_ds, val_ds, test_ds, weights_dict = load_dataset(
        config,
        args.dataset,
        args.model
    )
    print("Dataset loaded successfully!")

    # Build model
    print(f"\nBuilding {args.model} model...")
    num_segments = config['training'].get('num_segments', 3)
    model_builder = get_model_builder(args.model, num_segments)
    model = model_builder()
    print("Model built successfully!")

    # Train and evaluate
    output_name = f"{args.dataset}-{args.model}"
    if args.suffix:
        output_name += f"-{args.suffix}"

    print(f"\nStarting training for {output_name}...")
    print("Results will be saved to:")
    print(f"  - results-{output_name}.txt")
    print(f"  - accuracy-{output_name}.pdf")
    print(f"  - {output_name}final-model/")
    print()

    evaluate(output_name, train_ds, val_ds, test_ds, weights_dict, model=model)

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
