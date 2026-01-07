# EEG Foundation Challenge 2025
**Cross-task & Cross-subject EEG decoding**

This repository contains my implementation and experiments for the EEG Foundation Challenge 2025

## Background
Most EEG decoding models are trained on a single task and a small number of subjects, which limits their generalization ability.

The EEG Foundation Challenge aims to benchmark models that:
- generalize across **tasks**
- generalize across **subjects**
- learn transferable EEG representations at scale (HBN-EEG)

In this project, I focus on learning transferable EEG representations using self-supervised pretraining and fine-tuning for downstream regression tasks.

## Dataset
- **Dataset**: Healthy Brain Network EEG (HBN-EEG)
- **Channels**: 128
- **Sampling rate**: 100Hz
- **Subjects**: >3,000

### Tasks
- **Passive**: Resting State, Surround Suppression, Movie Watching
- **Active**: Contrast Change Detection, Sequence Learning, Symbol Search

### Data Split
- Train: Release 1-11 (except R5)
- Validation: Release 5
- Test: Release 12 (held-out, not accessible)

## Preprocessing

To reduce repeated I/O overhead and ensure reproducibility,
all EEG data were preprocessed and cached as Numpy arrays.

### Key preprocessing steps
- Load raw EEG using MNE
- Remove reference
- Downsample to 100 Hz
- Epoch into fixed-length windows
- Save as `.npy` files for fast loading

## Challenge 1


## Challenge 2

