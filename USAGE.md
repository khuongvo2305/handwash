# Unified Training Pipeline - Usage Guide

This guide explains how to use the new unified training pipeline that replaces the previous 9 redundant training scripts.

## Overview

The new pipeline consists of:
- **config/datasets.yaml** - Configuration for all datasets and training parameters
- **unified_dataset_loader.py** - Unified data loading for all datasets
- **train.py** - Single training script that replaces all 9 previous scripts

## Quick Start

### Basic Usage

```bash
# Train single-frame CNN on Kaggle dataset
python train.py --dataset kaggle --model frames

# Train video model with GRU on PSKUS dataset
python train.py --dataset pskus --model videos

# Train two-stream RGB+OF model on METC dataset
python train.py --dataset metc --model merged
```

### Available Options

**Datasets:**
- `kaggle` - Kaggle Hand Wash Dataset
- `pskus` - PSKUS Hospital Dataset
- `metc` - RSU METC Lab Dataset

**Models:**
- `frames` - Single-frame CNN (MobileNetV2/InceptionV3/Xception)
- `videos` - TimeDistributed CNN + GRU for video sequences
- `merged` - Two-stream network (RGB + Optical Flow)

## Configuration

### Using Custom Configuration

```bash
# Use custom config file
python train.py --dataset kaggle --model frames --config my_config.yaml
```

### Environment Variables

You can override configuration via environment variables:

```bash
# Change base model
export HANDWASH_NN="InceptionV3"

# Increase training epochs
export HANDWASH_NUM_EPOCHS=50

# Make more layers trainable
export HANDWASH_NUM_LAYERS=10

# Add extra dense layers
export HANDWASH_EXTRA_LAYERS=2

# Change number of frames for video model
export HANDWASH_NUM_FRAMES=10

# Disable mixed precision (if GPU doesn't support it)
export HANDWASH_MIXED_PRECISION=false

# Train the model
python train.py --dataset kaggle --model frames
```

### Editing config/datasets.yaml

The configuration file allows you to customize:

**Training Parameters:**
```yaml
training:
  batch_size: 32
  num_epochs: 20
  model_name: "MobileNetV2"  # or InceptionV3, Xception
  use_mixed_precision: true  # 2-3x speedup on modern GPUs
  cache_dataset: true        # Cache for faster training
```

**Dataset Paths:**
```yaml
datasets:
  kaggle:
    name: "Kaggle Hand Wash Dataset"
    fps: 30
    paths:
      frames_train: "path/to/kaggle/frames/trainval"
      frames_test: "path/to/kaggle/frames/test"
      of_train: "path/to/kaggle/of/trainval"
      of_test: "path/to/kaggle/of/test"
```

## Performance Optimizations

### Mixed Precision Training

**Enabled by default** for 2-3x speedup on modern GPUs (Volta, Turing, Ampere, etc.)

- Uses float16 for computation
- Uses float32 for variable storage
- Automatically handles loss scaling

To disable:
```bash
export HANDWASH_MIXED_PRECISION=false
python train.py --dataset kaggle --model frames
```

### Dataset Caching

Datasets are automatically cached in memory (if they fit) for faster training after the first epoch.

### Prefetching

Data is prefetched in parallel with training to maximize GPU utilization.

## Output Files

Training produces the following files:

```
results-{dataset}-{model}.txt          # Text results with accuracy and F1 scores
accuracy-{dataset}-{model}.pdf         # Training/validation accuracy plot
{dataset}-{model}final-model/          # Saved Keras model directory
```

Example:
```
results-kaggle-frames.txt
accuracy-kaggle-frames.pdf
kaggle-framesfinal-model/
```

## Examples

### Example 1: Quick Training

```bash
# Train on Kaggle dataset with single-frame model
python train.py --dataset kaggle --model frames
```

### Example 2: Custom Training

```bash
# Train on PSKUS with 50 epochs and InceptionV3
export HANDWASH_NUM_EPOCHS=50
export HANDWASH_NN="InceptionV3"
python train.py --dataset pskus --model frames
```

### Example 3: Fine-tuning

```bash
# Fine-tune the top 20 layers of MobileNetV2
export HANDWASH_NUM_LAYERS=20
export HANDWASH_NUM_EPOCHS=30
python train.py --dataset metc --model frames
```

### Example 4: Video Model

```bash
# Train video model with 10 frames
export HANDWASH_NUM_FRAMES=10
python train.py --dataset kaggle --model videos
```

### Example 5: Two-Stream Model

```bash
# Train RGB + Optical Flow model
# Note: Requires pre-computed optical flow
python train.py --dataset pskus --model merged
```

## Migration from Old Scripts

| Old Script | New Command |
|-----------|-------------|
| `kaggle-classify-frames.py` | `python train.py --dataset kaggle --model frames` |
| `kaggle-classify-videos.py` | `python train.py --dataset kaggle --model videos` |
| `kaggle-classify-merged-network.py` | `python train.py --dataset kaggle --model merged` |
| `pskus-classify-frames.py` | `python train.py --dataset pskus --model frames` |
| `pskus-classify-videos.py` | `python train.py --dataset pskus --model videos` |
| `pskus-classify-merged-network.py` | `python train.py --dataset pskus --model merged` |
| `rsu-metc-classify-frames.py` | `python train.py --dataset metc --model frames` |
| `rsu-metc-classify-videos.py` | `python train.py --dataset metc --model videos` |
| `rsu-metc-classify-merged-network.py` | `python train.py --dataset metc --model merged` |

## Troubleshooting

### Issue: Out of Memory

**Solution 1:** Reduce batch size in `config/datasets.yaml`:
```yaml
training:
  batch_size: 16  # or 8
```

**Solution 2:** Disable dataset caching:
```yaml
training:
  cache_dataset: false
```

### Issue: Mixed Precision Errors

If your GPU doesn't support mixed precision (pre-Volta):
```bash
export HANDWASH_MIXED_PRECISION=false
python train.py --dataset kaggle --model frames
```

### Issue: Dataset Not Found

Check the paths in `config/datasets.yaml` and ensure:
1. Datasets are downloaded
2. Preprocessing scripts have been run
3. Paths are correct (relative to project root)

### Issue: Optical Flow Not Found

For merged models, optical flow must be pre-computed:
```bash
python calculate-optical-flow.py dataset-kaggle/kaggle-dataset-6classes-preprocessed
```

## Advanced Usage

### Transfer Learning

Train on one dataset, then fine-tune on another:

```bash
# Train on Kaggle (large dataset)
export HANDWASH_NUM_EPOCHS=30
python train.py --dataset kaggle --model frames

# Use the model for transfer learning
export HANDWASH_PRETRAINED_MODEL="kaggle-framesfinal-model"
export HANDWASH_NUM_EPOCHS=20
python train.py --dataset pskus --model frames
```

### Experimentation

Add custom suffix to track experiments:

```bash
# Experiment 1: Baseline
python train.py --dataset kaggle --model frames --suffix baseline

# Experiment 2: More layers trainable
export HANDWASH_NUM_LAYERS=30
python train.py --dataset kaggle --model frames --suffix finetune30

# Experiment 3: With extra layers
export HANDWASH_EXTRA_LAYERS=2
python train.py --dataset kaggle --model frames --suffix extra2
```

Output files will include the suffix:
```
results-kaggle-frames-baseline.txt
results-kaggle-frames-finetune30.txt
results-kaggle-frames-extra2.txt
```

## Benefits of Unified Pipeline

1. **95% less code duplication** - 9 scripts → 1 script
2. **Consistent behavior** - Same logic for all datasets
3. **Easy configuration** - YAML file instead of editing code
4. **Better performance** - Mixed precision training enabled by default
5. **Easier maintenance** - Fix bugs once, applies everywhere
6. **Simpler experiments** - Just change config, no code edits needed

## Next Steps

1. See [CLAUDE.md](CLAUDE.md) for optimization recommendations
2. See [README.md](README.md) for dataset preparation instructions
3. Check the original papers for model architecture details

## Support

For issues or questions:
- Check this guide
- Review [CLAUDE.md](CLAUDE.md) for known issues
- Contact: atis.elsts@edi.lv
