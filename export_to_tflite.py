#!/usr/bin/env python3
"""
Unified TensorFlow Lite export script for all model types
Supports: frames, videos, merged, 3dcnn, tsn models
Includes quantization options for mobile deployment
"""

import argparse
import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Layer
from pathlib import Path


class MobileNetPreprocessingLayer(Layer):
    """Custom preprocessing layer for MobileNet models"""
    def __init__(self, **kwargs):
        super(MobileNetPreprocessingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MobileNetPreprocessingLayer, self).build(input_shape)

    def call(self, x):
        return (x / 127.5) - 1.0

    def compute_output_shape(self, input_shape):
        return input_shape


def load_model(model_path):
    """Load a Keras model with custom objects"""
    custom_objects = {"MobileNetPreprocessingLayer": MobileNetPreprocessingLayer}

    # Support both .h5 and SavedModel format
    if model_path.endswith('.h5'):
        print(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    else:
        # Assume it's a SavedModel directory
        print(f"Loading SavedModel from {model_path}")
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    return model


def convert_to_tflite(model, output_path, optimization='none', quantization='none'):
    """
    Convert Keras model to TFLite format

    Args:
        model: Keras model
        output_path: Path to save .tflite file
        optimization: 'none', 'default', 'size', 'latency'
        quantization: 'none', 'float16', 'int8', 'dynamic'
    """
    print(f"\nConverting model to TFLite...")
    print(f"Optimization: {optimization}")
    print(f"Quantization: {quantization}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Always enable both TFLite and TF ops for compatibility
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # TensorFlow Lite ops
        tf.lite.OpsSet.SELECT_TF_OPS      # TensorFlow ops
    ]

    # Apply optimization strategy
    if optimization == 'default':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("  - Applied default optimizations")

    elif optimization == 'size':
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        print("  - Optimized for size")

    elif optimization == 'latency':
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        print("  - Optimized for latency")

    # Apply quantization
    if quantization == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("  - Applied float16 quantization (reduces size by ~50%)")

    elif quantization == 'dynamic':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("  - Applied dynamic range quantization")

    elif quantization == 'int8':
        print("  - INT8 quantization requires a representative dataset")
        print("  - Falling back to dynamic quantization")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"\nError during conversion: {e}")
        print("\nTrying with experimental flag...")
        converter.experimental_new_converter = True
        tflite_model = converter.convert()

    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    # Report sizes
    model_size = len(tflite_model) / (1024 * 1024)
    print(f"\nConversion successful!")
    print(f"Output: {output_path}")
    print(f"Size: {model_size:.2f} MB")

    return tflite_model


def test_tflite_model(tflite_path, input_shape):
    """Test the TFLite model with random input"""
    print(f"\nTesting TFLite model...")

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")

    # Test with random input
    import numpy as np
    if len(input_details) == 1:
        # Single input (frames, videos, 3dcnn)
        test_input = np.random.rand(*input_details[0]['shape']).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
    else:
        # Multiple inputs (merged, tsn)
        for i, detail in enumerate(input_details):
            test_input = np.random.rand(*detail['shape']).astype(np.float32)
            interpreter.set_tensor(detail['index'], test_input)

    interpreter.invoke()

    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"Test successful! Output shape: {output.shape}")
    print(f"Output sample: {output[0, :5]}...")  # First 5 predictions

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Export Keras models to TensorFlow Lite format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python export_to_tflite.py kaggle-framesfinal-model

  # With float16 quantization (50% size reduction)
  python export_to_tflite.py kaggle-framesfinal-model --quantization float16

  # Optimize for size
  python export_to_tflite.py pskus-videosfinal-model --optimization size

  # Custom output path
  python export_to_tflite.py model.h5 --output my_model.tflite

  # Skip testing
  python export_to_tflite.py model --no-test
        """
    )

    parser.add_argument(
        'model_path',
        type=str,
        help='Path to Keras model (.h5 file or SavedModel directory)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for .tflite file (default: <model_name>.tflite)'
    )

    parser.add_argument(
        '--optimization',
        type=str,
        choices=['none', 'default', 'size', 'latency'],
        default='none',
        help='Optimization strategy (default: none)'
    )

    parser.add_argument(
        '--quantization',
        type=str,
        choices=['none', 'float16', 'dynamic', 'int8'],
        default='none',
        help='Quantization type (default: none). float16 recommended for mobile.'
    )

    parser.add_argument(
        '--no-test',
        action='store_true',
        help='Skip testing the converted model'
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found: {args.model_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        model_name = Path(args.model_path).stem
        if model_name.endswith('.h5'):
            model_name = model_name[:-3]
        output_path = f"{model_name}.tflite"

    print("="*60)
    print("TensorFlow Lite Export")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Output: {output_path}")
    print("="*60)

    # Load model
    try:
        model = load_model(args.model_path)
    except Exception as e:
        print(f"\nError loading model: {e}")
        sys.exit(1)

    print(f"Model loaded successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    # Convert to TFLite
    try:
        tflite_model = convert_to_tflite(
            model,
            output_path,
            args.optimization,
            args.quantization
        )
    except Exception as e:
        print(f"\nError during conversion: {e}")
        sys.exit(1)

    # Test the model
    if not args.no_test:
        try:
            test_tflite_model(output_path, model.input_shape)
        except Exception as e:
            print(f"\nWarning: Could not test model: {e}")
            print("The model was still saved successfully.")

    print("\n" + "="*60)
    print("Export completed successfully!")
    print("="*60)
    print(f"\nYou can now use {output_path} for mobile deployment")
    print("\nNext steps:")
    print("  1. Test the model on your target device")
    print("  2. Integrate with your mobile app")
    print("  3. Consider further optimizations if needed")


if __name__ == '__main__':
    main()
