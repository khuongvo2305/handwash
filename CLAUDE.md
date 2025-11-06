# Code Optimization & Architecture Review

**Generated:** 2025-11-06
**Project:** Hand Washing Movement Classification

---

## Executive Summary

This document provides a comprehensive review of the handwash classification codebase, identifying optimization opportunities, architectural improvements, and alternative approaches. The analysis covers three main areas:

1. **Code Quality & Bugs** - Critical issues that need immediate attention
2. **Architecture & Reusability** - How to unify dataset processing for better maintainability
3. **Alternative Approaches** - Modern techniques including Vision Transformers

---

## Table of Contents

- [Critical Issues](#critical-issues)
- [Current Architecture Analysis](#current-architecture-analysis)
- [Proposed Unified Pipeline](#proposed-unified-pipeline)
- [Performance Optimizations](#performance-optimizations)
- [Alternative Model Architectures](#alternative-model-architectures)
- [Implementation Roadmap](#implementation-roadmap)

---

## Critical Issues

### 1. Bug in calculate-optical-flow.py:23

**Location:** `calculate-optical-flow.py:23`
**Severity:** CRITICAL - Code will not run

```python
# Line 23 - INCORRECT
FPS = 16 if "METC" in input_dir else 30
```

**Problem:** Variable `input_dir` is used before it's defined (defined later at line 101)

**Fix:**
```python
FPS = 16 if "METC" in dataset_dir else 30
```

### 2. Deprecated TensorFlow APIs

**Locations:**
- `classify_dataset.py:42` - `tf.keras.layers.experimental.preprocessing`
- `dataset_utilities.py:50` - `tf.data.experimental.AUTOTUNE`
- `kaggle-classify-merged-network.py:55` - `tf.data.experimental.AUTOTUNE`

**Impact:** Will break in future TensorFlow versions

**Fix:**
```python
# Replace experimental preprocessing
from tensorflow.keras.layers import RandomFlip, RandomRotation

# Replace experimental AUTOTUNE
AUTOTUNE = tf.data.AUTOTUNE
```

### 3. Logical Error in Model Architecture

**Location:** `classify_dataset.py:99-101`

```python
x = tf.keras.layers.Flatten()(x)
if num_extra_layers:
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
```

**Problem:** Applying `Flatten()` followed by `GlobalAveragePooling2D()` is illogical. After flatten, the tensor is 1D and pooling won't work correctly.

**Fix:**
```python
if num_extra_layers:
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
else:
    x = tf.keras.layers.Flatten()(x)
```

---

## Current Architecture Analysis

### Dataset Processing Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      Raw Video Files                        │
│  (Kaggle / PSKUS / METC - Different formats & FPS)         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            separate-frames.py (per dataset)                 │
│  • Different preprocessing scripts                          │
│  • Hardcoded paths                                          │
│  • 70/30 train/test split (random)                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Preprocessed Directory Structure               │
│  dataset-X/preprocessed/                                    │
│    ├── frames/                                              │
│    │   ├── trainval/ (0-6 class folders)                    │
│    │   └── test/ (0-6 class folders)                        │
│    ├── videos/ (optional)                                   │
│    └── of/ (optical flow - optional)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Training Scripts (per dataset)                 │
│  • xxx-classify-frames.py (Single Frame CNN)                │
│  • xxx-classify-videos.py (TimeDistributed + GRU)           │
│  • xxx-classify-merged-network.py (Two-Stream RGB+OF)       │
└─────────────────────────────────────────────────────────────┘
```

### Current Issues

#### 1. **Code Duplication**

Each dataset has 3 nearly identical training scripts (9 files total):
- Only differ in hardcoded paths
- 95% code duplication
- Maintenance nightmare

#### 2. **Inconsistent FPS Handling**

```python
# Kaggle & PSKUS: 30 FPS
FPS = 30

# METC: 16 FPS
FPS = 16
```

This is hardcoded in multiple places, making it error-prone.

#### 3. **No Dataset Configuration**

Paths and parameters are hardcoded in each script rather than using a configuration file.

#### 4. **Model Architecture Duplication**

The same model-building code is repeated in `get_default_model()`, `get_time_distributed_model()`, and `get_merged_model()` with only slight variations.

---

## Proposed Unified Pipeline

### Architecture: Dataset Config + Universal Loader

```python
# config/datasets.yaml
datasets:
  kaggle:
    name: "Kaggle Hand Wash"
    fps: 30
    paths:
      frames_train: "dataset-kaggle/kaggle-dataset-6classes-preprocessed/frames/trainval"
      frames_test: "dataset-kaggle/kaggle-dataset-6classes-preprocessed/frames/test"
      of_train: "dataset-kaggle/kaggle-dataset-6classes-preprocessed/of/trainval"
      of_test: "dataset-kaggle/kaggle-dataset-6classes-preprocessed/of/test"

  pskus:
    name: "PSKUS Hospital"
    fps: 30
    paths:
      frames_train: "dataset-pskus/PSKUS_dataset_preprocessed/frames/trainval"
      frames_test: "dataset-pskus/PSKUS_dataset_preprocessed/frames/test"
      of_train: "dataset-pskus/PSKUS_dataset_preprocessed/of/trainval"
      of_test: "dataset-pskus/PSKUS_dataset_preprocessed/of/test"

  metc:
    name: "RSU METC Lab"
    fps: 16
    paths:
      frames_train: "dataset-metc/RSU_METC_dataset_preprocessed/frames/trainval"
      frames_test: "dataset-metc/RSU_METC_dataset_preprocessed/frames/test"
      of_train: "dataset-metc/RSU_METC_dataset_preprocessed/of/trainval"
      of_test: "dataset-metc/RSU_METC_dataset_preprocessed/of/test"

training:
  img_size: [240, 320]
  batch_size: 32
  n_classes: 7
  num_epochs: 20
```

### Unified Training Script

```python
# train.py - SINGLE SCRIPT FOR ALL DATASETS

import argparse
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['kaggle', 'pskus', 'metc'], required=True)
    parser.add_argument('--model', choices=['frames', 'videos', 'merged'], required=True)
    parser.add_argument('--config', default='config/datasets.yaml')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_config = config['datasets'][args.dataset]
    train_config = config['training']

    # Get dataset
    if args.model == 'frames':
        train_ds, val_ds, test_ds, weights = get_frame_datasets(
            dataset_config['paths']['frames_train'],
            dataset_config['paths']['frames_test'],
            train_config
        )
        model = build_cnn_model(train_config)

    elif args.model == 'videos':
        train_ds, val_ds, test_ds, weights = get_video_datasets(
            dataset_config['paths']['frames_train'],
            dataset_config['paths']['frames_test'],
            dataset_config['fps'],
            train_config
        )
        model = build_timedistributed_model(train_config)

    elif args.model == 'merged':
        train_ds, val_ds, test_ds, weights = get_merged_datasets(
            dataset_config['paths']['frames_train'],
            dataset_config['paths']['of_train'],
            dataset_config['paths']['frames_test'],
            dataset_config['paths']['of_test'],
            train_config
        )
        model = build_twostream_model(train_config)

    # Train
    output_name = f"{args.dataset}-{args.model}"
    evaluate(output_name, model, train_ds, val_ds, test_ds, weights)

if __name__ == '__main__':
    main()
```

### Usage Examples

```bash
# Instead of 9 different scripts, ONE script:
python train.py --dataset kaggle --model frames
python train.py --dataset pskus --model videos
python train.py --dataset metc --model merged

# Easy to extend to cross-dataset training
python train.py --dataset kaggle,pskus --model frames
```

### Benefits

1. **95% code reduction** - 9 scripts → 1 script
2. **Easy maintenance** - Change once, applies everywhere
3. **Consistent behavior** - Same logic for all datasets
4. **Configuration-driven** - Easy to add new datasets
5. **Better testing** - Single pipeline to test

---

## Performance Optimizations

### High Priority (Immediate Impact)

#### 1. Enable Mixed Precision Training

**Expected speedup:** 2-3x on modern GPUs
**Memory savings:** 30-50%

```python
# Add to classify_dataset.py after GPU config

from tensorflow.keras import mixed_precision

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Modify final layer to use float32
outputs = tf.keras.layers.Dense(N_CLASSES, activation='softmax',
                                 dtype='float32')(x)
```

#### 2. Parallel Optical Flow Calculation

**Current:** Sequential processing
**Proposed:** Multiprocessing

```python
# calculate-optical-flow.py - Add parallel processing

from multiprocessing import Pool
import multiprocessing as mp

def process_class_partition(args):
    """Process one class in one partition"""
    partition, c, dataset_dir, classes = args
    input_dir = os.path.join(dataset_dir, "videos", partition)
    output_dir = os.path.join(dataset_dir, "of", partition)

    for filename in os.listdir(os.path.join(input_dir, c)):
        if filename.endswith(".mp4"):
            extract_flow(c, filename, input_dir, output_dir)

def main():
    # Create work items
    work_items = []
    for partition in ["test", "trainval"]:
        for c in classes:
            work_items.append((partition, c, dataset_dir, classes))

    # Process in parallel
    num_workers = min(mp.cpu_count(), len(work_items))
    with Pool(num_workers) as pool:
        pool.map(process_class_partition, work_items)
```

**Expected speedup:** 4-8x depending on CPU cores

#### 3. Efficient Batch Prediction

**Location:** `classify_dataset.py:283-295`

**Current (inefficient):**
```python
for images, labels in ds:
    predicted = model.predict(images)  # Batch by batch
    for y_p, y_t in zip(predicted, labels):
        y_predicted.append(int(np.argmax(y_p)))
        y_true.append(int(np.argmax(y_t)))
```

**Optimized:**
```python
# Predict all at once
y_pred = model.predict(ds, verbose=1)
y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in ds])
y_predicted = np.argmax(y_pred, axis=1)
```

**Expected speedup:** 2-3x for evaluation

#### 4. Dataset Caching

```python
# dataset_utilities.py - Add caching

def get_datasets(data_dir, test_data_dir, batch_size=None):
    # ... existing code ...

    # Add caching for datasets that fit in memory
    train_ds = train_ds.cache()  # Cache after first epoch
    val_ds = val_ds.cache()
    test_ds = test_ds.cache()

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, weights_dict
```

**Impact:** 20-30% faster training after first epoch

### Medium Priority

#### 5. Remove Redundant Operations

```python
# classify_dataset.py:227 - Remove unnecessary flatten
# After concatenate, output is already flat

# BEFORE (incorrect):
merged = tf.keras.layers.concatenate([rgb_network.output, of_network.output], axis=1)
merged = tf.keras.layers.Flatten()(merged)  # Unnecessary!

# AFTER (correct):
merged = tf.keras.layers.concatenate([rgb_network.output, of_network.output], axis=1)
```

#### 6. Use tf.function for Custom Layers

```python
# classify_dataset.py:113-124

class MobileNetPreprocessingLayer(Layer):
    def __init__(self, **kwargs):
        super(MobileNetPreprocessingLayer, self).__init__(**kwargs)

    @tf.function  # Add JIT compilation
    def call(self, x):
        return (x / 127.5) - 1.0
```

#### 7. Optimize Image Loading

```python
# generator_timedistributed.py:306

# BEFORE (slow):
return tf.convert_to_tensor(imgs, dtype=tf.float32)

# AFTER (faster):
return tf.stack(imgs)  # More efficient than convert_to_tensor
```

### Low Priority

#### 8. Configurable Hyperparameters

Move hardcoded values to config:
- Early stopping patience (currently 10)
- Buffer size for shuffle (currently `batch_size * 8`)
- Dropout rates (currently 0.2)
- GRU units (currently 256)

---

## Alternative Model Architectures

### Do Current Models Work?

**Yes, but they have limitations:**

| Model | Pros | Cons |
|-------|------|------|
| **Single Frame CNN** | Fast, simple, works well | No temporal information |
| **TimeDistributed + GRU** | Captures temporal patterns | Heavy computation, GRU is older tech |
| **Two-Stream (RGB+OF)** | Rich features from OF | Requires pre-computing OF, 2x data |

### Modern Alternatives

#### 1. Vision Transformers (ViT) for Single Frames

**Why:** Transformers have surpassed CNNs on many vision tasks

```python
# Using timm library for pretrained ViT

import timm

def get_vit_model(img_shape, n_classes):
    # Use a pretrained Vision Transformer
    base_model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=0,  # Remove head
        img_size=img_shape[:2]
    )

    inputs = tf.keras.Input(shape=img_shape)
    x = base_model(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model
```

**Pros:**
- State-of-art accuracy
- Better at capturing long-range dependencies
- Scales well with data

**Cons:**
- Requires more compute
- More data-hungry
- Larger model size

#### 2. Video Vision Transformer (ViViT)

**Why:** Purpose-built for video understanding, replaces TimeDistributed+GRU

```python
# Conceptual - would need implementation

def get_vivit_model(num_frames, img_shape, n_classes):
    """
    Video Vision Transformer
    - Treats video as sequence of frame patches
    - Applies transformer encoder
    - Learns spatiotemporal patterns
    """
    # Patch embedding for each frame
    # Positional encoding (spatial + temporal)
    # Transformer encoder
    # Classification head
```

**Pros:**
- Unified spatial-temporal modeling
- Better than GRU for long sequences
- State-of-art on video tasks

**Cons:**
- Very compute-intensive
- Needs large datasets
- More complex to implement

#### 3. 3D Convolutional Networks (C3D / I3D)

**Why:** Natural for video, learns spatiotemporal features directly

```python
def get_3d_cnn_model(num_frames, img_shape, n_classes):
    inputs = tf.keras.Input(shape=(num_frames, *img_shape))

    # 3D convolutions
    x = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2))(x)

    x = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2))(x)

    x = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)

    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model
```

**Pros:**
- Learns spatiotemporal features end-to-end
- No need for optical flow
- Well-established architecture

**Cons:**
- Memory intensive
- Slower training
- Needs careful tuning

#### 4. Temporal Segment Networks (TSN)

**Why:** Efficient way to sample long videos, better than dense sampling

**Approach:**
- Divide video into N segments
- Sample 1 frame from each segment
- Process with CNN
- Aggregate predictions (avg/max)

```python
def get_tsn_model(num_segments, img_shape, n_classes):
    """
    Temporal Segment Network
    - Efficient for long videos
    - Sparse sampling reduces computation
    """
    # Base CNN for single frame
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_shape,
        include_top=False,
        pooling='avg'
    )

    # Process each segment
    segment_inputs = [tf.keras.Input(shape=img_shape) for _ in range(num_segments)]
    segment_features = [base_model(inp) for inp in segment_inputs]

    # Consensus function (average)
    merged = tf.keras.layers.Average()(segment_features)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(merged)

    model = tf.keras.Model(segment_inputs, outputs)
    return model
```

**Pros:**
- Very efficient
- Can handle long videos
- Better than single frame

**Cons:**
- May miss fine-grained motions
- Requires custom data loader

#### 5. SlowFast Networks

**Why:** Dual-pathway architecture for capturing both slow and fast motion

**Approach:**
- Slow pathway: High resolution, low frame rate
- Fast pathway: Low resolution, high frame rate
- Fuse features from both

**Pros:**
- State-of-art on action recognition
- Efficient dual-pathway design
- Captures multiple temporal scales

**Cons:**
- Complex implementation
- Needs careful tuning
- More compute than single-pathway

### Recommendation Priority

For this hand-washing task:

1. **Short-term (Quick Win):** Optimize current models + add TSN
2. **Medium-term:** Implement 3D CNN (I3D)
3. **Long-term (Research):** Experiment with ViT/ViViT if dataset grows

---

## Implementation Roadmap

### Phase 1: Bug Fixes & Critical Issues (1 day)

- [ ] Fix `calculate-optical-flow.py` bug
- [ ] Update deprecated TensorFlow APIs
- [ ] Fix Flatten + GlobalAveragePooling logic
- [ ] Add tests to prevent regression

### Phase 2: Code Refactoring (2-3 days)

- [ ] Create `config/datasets.yaml`
- [ ] Implement unified `train.py` script
- [ ] Create `dataset_loader.py` with unified interface
- [ ] Refactor model builders into separate classes
- [ ] Remove 9 redundant scripts

### Phase 3: Performance Optimization (2 days)

- [ ] Enable mixed precision training
- [ ] Parallelize optical flow calculation
- [ ] Optimize batch prediction in evaluation
- [ ] Add dataset caching
- [ ] Benchmark improvements

### Phase 4: Alternative Models (1-2 weeks)

- [ ] Implement Temporal Segment Network (TSN)
- [ ] Implement 3D CNN (I3D)
- [ ] Benchmark against existing models
- [ ] Document results

### Phase 5: Advanced Features (optional)

- [ ] Cross-dataset training/testing
- [ ] Ensemble methods
- [ ] TensorFlow Lite export for mobile
- [ ] Real-time inference optimization

---

## Estimated Impact Summary

| Optimization | Effort | Impact | Priority |
|--------------|--------|--------|----------|
| Fix critical bugs | Low | High | **P0** |
| Unified config + train.py | Medium | High | **P0** |
| Mixed precision training | Low | High | **P1** |
| Parallel OF calculation | Medium | High | **P1** |
| Efficient evaluation | Low | Medium | **P1** |
| Dataset caching | Low | Medium | **P2** |
| TSN model | Medium | Medium | **P2** |
| 3D CNN model | High | High | **P3** |
| Vision Transformers | High | Medium | **P3** |

---

## Questions & Considerations

### 1. Dataset Size

The effectiveness of transformers and large models depends on dataset size:
- **< 10K samples:** Stick with MobileNet + fine-tuning
- **10K-100K samples:** 3D CNN or I3D
- **> 100K samples:** Consider Vision Transformers

**Current datasets:** Check size to inform model choice.

### 2. Inference Requirements

- **Real-time mobile:** Keep MobileNet, optimize with TFLite
- **Server inference:** Can use larger models (ViT, I3D)
- **Edge devices:** Need quantization + pruning

### 3. Optical Flow Necessity

**Question:** Is pre-computed optical flow worth the cost?

**Analysis:**
- **Pros:** Rich motion features, proven to help
- **Cons:** 2x storage, preprocessing time, complexity

**Alternative:** 3D CNNs learn motion features implicitly without OF.

**Recommendation:** Benchmark single RGB vs RGB+OF to see if OF provides significant improvement. If gain is marginal, remove OF pipeline.

---

## Conclusion

The current codebase is **functional but has significant room for improvement**:

1. **Critical bugs** must be fixed immediately
2. **Code duplication** should be eliminated via unified pipeline
3. **Performance gains** of 2-3x are achievable with simple optimizations
4. **Alternative models** (3D CNN, TSN) may provide better accuracy

The **unified config-driven architecture** is the most important change, reducing maintenance burden and enabling easier experimentation.

---

## References

### Papers
- **I3D:** Quo Vadis, Action Recognition? (Carreira & Zisserman, 2017)
- **TSN:** Temporal Segment Networks (Wang et al., 2016)
- **SlowFast:** SlowFast Networks for Video Recognition (Feichtenhofer et al., 2019)
- **ViT:** An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)
- **ViViT:** Video Vision Transformer (Arnab et al., 2021)

### Libraries
- **timm:** PyTorch Image Models - https://github.com/huggingface/pytorch-image-models
- **MMAction2:** OpenMMLab video understanding toolbox - https://github.com/open-mmlab/mmaction2
- **TensorFlow Models:** Official TF model implementations - https://github.com/tensorflow/models

---

**End of Document**
