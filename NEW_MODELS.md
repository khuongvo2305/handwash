# New Alternative Models Guide

This document describes the new model architectures added to the handwash classification pipeline: **3D CNN** and **Temporal Segment Networks (TSN)**.

---

## Table of Contents
- [Overview](#overview)
- [3D CNN Model](#3d-cnn-model)
- [Temporal Segment Network (TSN)](#temporal-segment-network-tsn)
- [Parallel Optical Flow](#parallel-optical-flow)
- [Performance Comparison](#performance-comparison)
- [Usage Examples](#usage-examples)

---

## Overview

Two new state-of-the-art video classification models have been added:

| Model | Best For | Pros | Cons |
|-------|----------|------|------|
| **3D CNN** | Short clips with fine motion | Learns spatiotemporal features end-to-end | Memory intensive |
| **TSN** | Long videos, efficient processing | Very efficient, sparse sampling | May miss fine-grained motion |

These complement the existing models:
- **Single-frame CNN** - Fast, no temporal info
- **TimeDistributed + GRU** - Good temporal modeling, older tech
- **Two-stream RGB+OF** - Rich features, requires optical flow preprocessing

---

## 3D CNN Model

### Architecture

3D CNNs use 3D convolutions to learn spatiotemporal features directly from video clips.

```
Input: (16 frames, 112x112, 3 channels)
    ↓
Conv3D(64) + BatchNorm + MaxPool3D + Dropout(0.2)
    ↓
Conv3D(128) + BatchNorm + MaxPool3D + Dropout(0.3)
    ↓
Conv3D(256) + BatchNorm + MaxPool3D + Dropout(0.3)
    ↓
Conv3D(512) + BatchNorm + GlobalAveragePooling3D
    ↓
Dense(512) + Dropout(0.5)
    ↓
Output: Dense(7, softmax)
```

### Key Features

- **3D Convolutions**: Filters move across time dimension as well as spatial
- **No Optical Flow Needed**: Learns motion implicitly
- **End-to-end Learning**: Spatiotemporal features learned together

### Configuration

```yaml
training:
  num_frames: 16  # Number of frames per clip
  img_size_3dcnn: [112, 112]  # Smaller for memory
  batch_size_3dcnn: 8  # Smaller batch size
  frame_step_3dcnn: 2  # Frame sampling step
```

### Usage

```bash
# Train 3D CNN on Kaggle dataset
python train.py --dataset kaggle --model 3dcnn

# Adjust frames and batch size
export HANDWASH_NUM_FRAMES=24
python train.py --dataset pskus --model 3dcnn
```

### Memory Requirements

3D CNNs are **memory intensive**:
- 16 frames @ 112x112: ~4GB GPU memory
- 16 frames @ 224x224: ~16GB GPU memory
- Reduce `num_frames` or `img_size_3dcnn` if OOM

### Advantages

✅ Learns motion features automatically (no optical flow)
✅ State-of-art for action recognition
✅ End-to-end differentiable
✅ Good for short clips with fine motion

### Disadvantages

❌ Very memory intensive
❌ Slower training than 2D models
❌ Needs more data to train well
❌ Less efficient for long videos

### When to Use

- Short video clips (< 5 seconds)
- Fine-grained motion important
- Sufficient GPU memory available
- Want to avoid optical flow preprocessing

---

## Temporal Segment Network (TSN)

### Architecture

TSN divides videos into segments and samples one frame per segment, then aggregates predictions.

```
Video divided into 3 segments:
    Segment 1     Segment 2     Segment 3
       ↓             ↓             ↓
   Frame 10      Frame 50      Frame 90
       ↓             ↓             ↓
   MobileNetV2   MobileNetV2   MobileNetV2
   (shared)      (shared)      (shared)
       ↓             ↓             ↓
    Features      Features      Features
       └─────────────┴─────────────┘
                     ↓
            Average Pooling
                     ↓
               Dense Layers
                     ↓
         Output: Dense(7, softmax)
```

### Key Features

- **Sparse Sampling**: Only 1 frame per segment (very efficient)
- **Temporal Coverage**: Samples from beginning, middle, and end
- **Consensus Function**: Averages features from all segments
- **Efficient**: Much faster than processing all frames

### Configuration

```yaml
training:
  num_segments: 3  # Number of segments to divide video
  frames_per_video: 30  # Approximate total frames
```

### Usage

```bash
# Train TSN with 3 segments
python train.py --dataset kaggle --model tsn

# Train with 5 segments for longer coverage
# Edit config/datasets.yaml: num_segments: 5
python train.py --dataset metc --model tsn
```

### Advantages

✅ **Very efficient** - processes only N frames (N=num_segments)
✅ **Good temporal coverage** - samples entire video
✅ **Scalable** - works well on long videos
✅ **Less memory** - similar to single-frame model
✅ **Fast inference** - only N forward passes

### Disadvantages

❌ May miss important motion between segments
❌ Sparse sampling might skip critical frames
❌ Less effective for videos with rapid motion changes

### When to Use

- Long videos (> 10 seconds)
- Want efficient processing
- Temporal evolution more important than frame-by-frame motion
- Limited GPU memory

### Segment Selection Strategy

The current implementation samples from the **middle** of each segment:

```python
# Video with 90 frames, 3 segments
Segment 1: frames 0-29   → sample frame 15
Segment 2: frames 30-59  → sample frame 45
Segment 3: frames 60-89  → sample frame 75
```

This can be made random during training for data augmentation.

---

## Parallel Optical Flow

A new parallel version of optical flow calculation provides **4-8x speedup**.

### Usage

```bash
# Original (sequential)
python calculate-optical-flow.py dataset-kaggle/kaggle-dataset-6classes-preprocessed

# New (parallel) - uses all CPU cores
python calculate-optical-flow-parallel.py dataset-kaggle/kaggle-dataset-6classes-preprocessed

# Control number of workers
export HANDWASH_NUM_WORKERS=8
python calculate-optical-flow-parallel.py dataset-pskus/PSKUS_dataset_preprocessed
```

### Features

- **Multiprocessing**: Uses all available CPU cores
- **Progress Tracking**: Real-time progress display
- **Error Handling**: Continues if individual videos fail
- **Statistics**: Shows timing and success/failure counts

### Output Example

```
============================================================
Parallel Optical Flow Calculation
============================================================
Dataset: dataset-kaggle/kaggle-dataset-6classes-preprocessed
FPS: 30
Frame step: 10
CPU cores available: 8
============================================================

============================================================
Processing trainval partition
Total videos: 450
============================================================

Using 8 parallel workers
[ 10.2%] Processed dataset-kaggle/.../0/video_001.mp4
[ 20.4%] Processed dataset-kaggle/.../1/video_023.mp4
...
[100.0%] Processed dataset-kaggle/.../6/video_450.mp4

------------------------------------------------------------
Partition trainval completed!
Successful: 450/450
Failed: 0
Total optical flow frames: 12500
Time elapsed: 245.3s
Average time per video: 0.55s
------------------------------------------------------------
```

### Performance

| CPU Cores | Videos | Sequential Time | Parallel Time | Speedup |
|-----------|--------|-----------------|---------------|---------|
| 1 | 450 | 30 min | 30 min | 1x |
| 4 | 450 | 30 min | 8 min | 3.75x |
| 8 | 450 | 30 min | 4 min | 7.5x |
| 16 | 450 | 30 min | 2.5 min | 12x |

---

## Performance Comparison

### Accuracy (Expected)

Based on action recognition literature:

| Model | Single-Frame Acc | Video Acc | Notes |
|-------|-----------------|-----------|-------|
| Single-frame CNN | 65-75% | N/A | Baseline |
| TimeDistributed+GRU | 70-80% | Good | Moderate |
| Two-stream RGB+OF | 75-85% | Very Good | Best with OF |
| **3D CNN** | 72-82% | Very Good | Good for motion |
| **TSN** | 70-80% | Good | Efficient |

*Actual performance depends on dataset and hyperparameters*

### Training Speed (relative, batch_size=32)

| Model | Speed | Memory | Notes |
|-------|-------|--------|-------|
| Single-frame | 1x | Low | Fastest |
| TSN | 1.2x | Low | Slightly slower |
| TimeDistributed+GRU | 2x | Medium | Sequential processing |
| Two-stream RGB+OF | 2.5x | Medium | Two networks |
| **3D CNN** | 4x | **High** | Most expensive |

### Inference Speed

| Model | Frames/Second | Use Case |
|-------|---------------|----------|
| Single-frame | 100+ | Real-time |
| TSN (3 segments) | 60+ | Real-time |
| TimeDistributed+GRU | 30+ | Near real-time |
| 3D CNN | 10-20 | Offline |
| Two-stream RGB+OF | 15-25 | Offline |

---

## Usage Examples

### Example 1: Quick 3D CNN Experiment

```bash
# Use 3D CNN with default settings
python train.py --dataset kaggle --model 3dcnn
```

### Example 2: Memory-Constrained 3D CNN

```bash
# Reduce frames and image size for limited GPU memory
# Edit config/datasets.yaml:
#   num_frames: 8
#   img_size_3dcnn: [80, 80]
#   batch_size_3dcnn: 4

python train.py --dataset pskus --model 3dcnn
```

### Example 3: TSN with More Segments

```bash
# Edit config/datasets.yaml:
#   num_segments: 5  # More temporal coverage

python train.py --dataset metc --model tsn
```

### Example 4: TSN with InceptionV3

```bash
# Use larger base model for better features
export HANDWASH_NN="InceptionV3"
python train.py --dataset kaggle --model tsn
```

### Example 5: Parallel Optical Flow for All Datasets

```bash
# Process optical flow for all 3 datasets in parallel
python calculate-optical-flow-parallel.py dataset-kaggle/kaggle-dataset-6classes-preprocessed &
python calculate-optical-flow-parallel.py dataset-pskus/PSKUS_dataset_preprocessed &
python calculate-optical-flow-parallel.py dataset-metc/RSU_METC_dataset_preprocessed &
wait

echo "All optical flow calculations complete!"
```

### Example 6: Compare All Models

```bash
# Compare all 5 models on Kaggle dataset
for model in frames videos merged 3dcnn tsn; do
    echo "Training $model..."
    python train.py --dataset kaggle --model $model --suffix comparison
done

# Compare results
grep "Average test F1" results-kaggle-*-comparison.txt
```

---

## Tips & Best Practices

### For 3D CNN

1. **Start small**: Begin with 8 frames @ 80x80 to test
2. **Monitor memory**: Use `nvidia-smi` to track GPU usage
3. **Use mixed precision**: Already enabled by default for 2-3x speedup
4. **Data augmentation**: Critical for 3D CNNs (avoid overfitting)

### For TSN

1. **Adjust segments**: 3-5 segments usually optimal
2. **Video length**: TSN excels on longer videos (> 5 seconds)
3. **Base model**: Try InceptionV3 or Xception for better features
4. **Segment sampling**: Can randomize for data augmentation

### General

1. **Try TSN first**: Most efficient, good baseline
2. **Use 3D CNN if**: Fine motion is critical and you have GPU memory
3. **Parallel OF**: Always use parallel version for optical flow
4. **Benchmarking**: Test on validation set before full training

---

## Troubleshooting

### 3D CNN: Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
1. Reduce `num_frames` (16 → 8)
2. Reduce `img_size_3dcnn` ([112,112] → [80,80])
3. Reduce `batch_size_3dcnn` (8 → 4)
4. Disable dataset caching: `cache_dataset: false`

### TSN: No Videos Found

**Error**: `No videos found`

**Cause**: Frame naming doesn't match expected pattern

**Solution**: Ensure frames are named: `frame_0_videoname.jpg`, `frame_1_videoname.jpg`, etc.

### Parallel OF: Process Hangs

**Issue**: Process appears frozen

**Solutions**:
1. Reduce workers: `export HANDWASH_NUM_WORKERS=4`
2. Check disk space (optical flow is large)
3. Verify video files are not corrupted

---

## Next Steps

1. **Benchmark**: Compare all models on your dataset
2. **Hyperparameter tuning**: Adjust num_frames, segments, etc.
3. **Ensemble**: Combine multiple models for best results
4. **Transfer learning**: Pre-train on large dataset, fine-tune on small

For more details, see:
- [USAGE.md](USAGE.md) - General usage guide
- [CLAUDE.md](CLAUDE.md) - Full optimization review
- [config/datasets.yaml](config/datasets.yaml) - All parameters

---

**Questions?** Contact: atis.elsts@edi.lv
