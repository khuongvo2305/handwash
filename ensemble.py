#!/usr/bin/env python3
"""
Ensemble prediction system for combining multiple models
Supports voting, averaging, and weighted averaging strategies
"""

import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
from tensorflow.keras.layers import Layer


class MobileNetPreprocessingLayer(Layer):
    """Custom preprocessing layer for MobileNet models"""
    def __init__(self, **kwargs):
        super(MobileNetPreprocessingLayer, self).__init__(**kwargs)

    def call(self, x):
        return (x / 127.5) - 1.0


def load_model(model_path):
    """Load a single model"""
    custom_objects = {"MobileNetPreprocessingLayer": MobileNetPreprocessingLayer}

    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        return model
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None


def load_models(model_paths):
    """Load multiple models"""
    models = []
    for path in model_paths:
        print(f"Loading {path}...")
        model = load_model(path)
        if model is not None:
            models.append((Path(path).stem, model))

    if not models:
        raise ValueError("No models could be loaded!")

    print(f"\nSuccessfully loaded {len(models)} models")
    return models


def ensemble_predict(models, dataset, strategy='average', weights=None):
    """
    Make ensemble predictions

    Args:
        models: List of (name, model) tuples
        dataset: TensorFlow dataset
        strategy: 'vote', 'average', 'weighted'
        weights: List of weights for 'weighted' strategy

    Returns:
        predictions, true_labels
    """
    print(f"\nEnsemble strategy: {strategy}")

    if strategy == 'weighted' and weights is None:
        raise ValueError("Weights must be provided for weighted strategy")

    if strategy == 'weighted' and len(weights) != len(models):
        raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(models)})")

    # Normalize weights
    if weights is not None:
        weights = np.array(weights) / np.sum(weights)
        print(f"Normalized weights: {weights}")

    all_predictions = []
    y_true = []

    # Get predictions from each model
    for name, model in models:
        print(f"  Getting predictions from {name}...")
        preds = []
        labels = []

        for images, batch_labels in dataset:
            batch_preds = model.predict(images, verbose=0)
            preds.append(batch_preds)
            labels.append(batch_labels.numpy())

        preds = np.vstack(preds)
        all_predictions.append(preds)

        if not y_true:  # Only store once
            y_true = np.vstack(labels)

    all_predictions = np.array(all_predictions)  # Shape: (n_models, n_samples, n_classes)
    print(f"  Predictions shape: {all_predictions.shape}")

    # Apply ensemble strategy
    if strategy == 'vote':
        # Hard voting: take argmax for each model, then vote
        votes = np.argmax(all_predictions, axis=2)  # (n_models, n_samples)
        ensemble_preds = np.array([
            np.bincount(votes[:, i], minlength=all_predictions.shape[2]).argmax()
            for i in range(votes.shape[1])
        ])

    elif strategy == 'average':
        # Soft voting: average probabilities
        avg_probs = np.mean(all_predictions, axis=0)  # (n_samples, n_classes)
        ensemble_preds = np.argmax(avg_probs, axis=1)

    elif strategy == 'weighted':
        # Weighted average of probabilities
        weighted_probs = np.tensordot(weights, all_predictions, axes=([0], [0]))
        ensemble_preds = np.argmax(weighted_probs, axis=1)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    y_true_labels = np.argmax(y_true, axis=1)

    return ensemble_preds, y_true_labels


def evaluate_ensemble(ensemble_preds, y_true, n_classes=7):
    """Evaluate ensemble predictions"""
    accuracy = np.mean(ensemble_preds == y_true)

    # Confusion matrix
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for pred, true in zip(ensemble_preds, y_true):
        matrix[true, pred] += 1

    # Per-class metrics
    f1_scores = []
    for i in range(n_classes):
        true_pos = matrix[i, i]
        false_pos = np.sum(matrix[:, i]) - true_pos
        false_neg = np.sum(matrix[i, :]) - true_pos

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    return accuracy, f1_scores, matrix


def main():
    parser = argparse.ArgumentParser(
        description='Ensemble prediction for hand washing classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Average ensemble of 3 models
  python ensemble.py --models model1 model2 model3 --dataset kaggle --strategy average

  # Weighted ensemble
  python ensemble.py --models model1 model2 model3 --dataset kaggle \\
      --strategy weighted --weights 0.5 0.3 0.2

  # Voting ensemble
  python ensemble.py --models model1 model2 model3 --dataset pskus --strategy vote

  # Save predictions
  python ensemble.py --models model1 model2 --dataset metc --output predictions.json
        """
    )

    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='Paths to model files or directories'
    )

    parser.add_argument(
        '--dataset',
        choices=['kaggle', 'pskus', 'metc'],
        required=True,
        help='Dataset to evaluate on'
    )

    parser.add_argument(
        '--strategy',
        choices=['vote', 'average', 'weighted'],
        default='average',
        help='Ensemble strategy (default: average)'
    )

    parser.add_argument(
        '--weights',
        nargs='+',
        type=float,
        default=None,
        help='Weights for weighted strategy (must sum to 1.0)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Save predictions to JSON file'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/datasets.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    print("="*60)
    print("Ensemble Prediction System")
    print("="*60)
    print(f"Models: {len(args.models)}")
    for model in args.models:
        print(f"  - {model}")
    print(f"Dataset: {args.dataset}")
    print(f"Strategy: {args.strategy}")
    print("="*60)

    # Load models
    models = load_models(args.models)

    # Load dataset
    print(f"\nLoading {args.dataset} test dataset...")
    # Import here to avoid circular dependency
    from unified_dataset_loader import UnifiedDatasetLoader
    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)

    loader = UnifiedDatasetLoader(config)

    # Load appropriate dataset for first model type
    # (Assuming all models use same input format)
    dataset_config = config['datasets'][args.dataset]
    test_data_dir = dataset_config['paths']['frames_test']

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        seed=123,
        image_size=(config['training']['img_height'], config['training']['img_width']),
        label_mode='categorical',
        batch_size=config['training']['batch_size']
    )

    print(f"Test dataset loaded: {len(list(test_ds))} batches")

    # Make ensemble predictions
    ensemble_preds, y_true = ensemble_predict(
        models, test_ds, args.strategy, args.weights
    )

    # Evaluate
    accuracy, f1_scores, matrix = evaluate_ensemble(ensemble_preds, y_true)

    # Print results
    print("\n" + "="*60)
    print("Ensemble Results")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f} ({100*accuracy:.2f}%)")
    print(f"\nPer-class F1 scores:")
    for i, f1 in enumerate(f1_scores):
        print(f"  Class {i}: {f1:.4f}")
    print(f"\nMean F1 score: {np.mean(f1_scores):.4f}")

    print(f"\nConfusion Matrix:")
    print(matrix)

    # Compare with individual models
    print("\n" + "-"*60)
    print("Individual Model Performance:")
    print("-"*60)

    for i, (name, model) in enumerate(models):
        preds = []
        for images, _ in test_ds:
            batch_preds = model.predict(images, verbose=0)
            preds.append(np.argmax(batch_preds, axis=1))
        preds = np.concatenate(preds)
        individual_acc = np.mean(preds == y_true)
        print(f"{name}: {100*individual_acc:.2f}%")

    print(f"\nEnsemble ({args.strategy}): {100*accuracy:.2f}%")
    improvement = accuracy - max([np.mean(preds == y_true) for preds, _, _ in [
        (np.concatenate([np.argmax(model.predict(images, verbose=0), axis=1) for images, _ in test_ds]), None, None)
        for _, model in models
    ]])
    if improvement > 0:
        print(f"Improvement: +{100*improvement:.2f}%")

    # Save predictions
    if args.output:
        results = {
            'models': [name for name, _ in models],
            'strategy': args.strategy,
            'weights': weights.tolist() if args.strategy == 'weighted' else None,
            'accuracy': float(accuracy),
            'f1_scores': [float(f1) for f1 in f1_scores],
            'mean_f1': float(np.mean(f1_scores)),
            'confusion_matrix': matrix.tolist(),
            'predictions': ensemble_preds.tolist(),
            'true_labels': y_true.tolist()
        }

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {args.output}")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()
