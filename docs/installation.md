# Installation Guide

This document provides step-by-step instructions to set up FractalSERL on your system. The installation includes setting up the Conda environment, installing JAX with GPU support, and installing the core packages.

## Prerequisites

### Hardware Requirements

For optimal performance and to reproduce the experiments in our paper, we recommend the following system configuration:

- **Processor:** AMD Ryzen Threadripper 1950x or equivalent (16+ cores recommended)
- **RAM:** 128 GB
- **GPU:** NVIDIA RTX 4070 (12 GB VRAM) or better
> **Note:** Baseline experiments were conducted with an RTX 4090 (82.6 FP32 TFLOPS, 1008 GB/s bandwidth). The RTX 4070 has approximately 2.84× lower FP32 compute (29.1 TFLOPS) and 2× lower bandwidth (504 GB/s), making our results on RTX 4070 all the more significant.

### Software Requirements

- **Operating System:** Ubuntu 20.04 LTS
- Python 3.10
- Conda (Miniconda or Anaconda)
- CUDA Toolkit 12.0+ (for GPU support)

> ⚠️ **Note on End-of-Life Support:** Ubuntu 20.04 LTS reaches end-of-standard support in April 2025. If using Ubuntu 20.04 for real-robot applications requiring ROS1, note that ROS1 is also in end-of-life status. For new deployments, consider upgrading to Ubuntu 22.04 LTS with ROS2.

## Installation Steps

### 1. Setup Conda Environment

Create a new Conda environment with Python 3.10:

```bash
conda create -n serl python=3.10
conda activate serl
```

### 2. Install JAX

Choose the installation method based on your hardware:

#### For GPU (Recommended for RTX 4070 / 4090):

```bash
pip install --upgrade "jax[cuda12]==0.6.2"
```

#### For CPU (Not Recommended):

```bash
pip install --upgrade "jax[cpu]"
```

#### For TPU:

```bash
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

For more details on JAX installation, see the [JAX GitHub page](https://github.com/google/jax).

### 3. Install serl_launcher

Navigate to the `serl_launcher` directory and install:

```bash
cd serl_launcher
pip install -e .
pip install -r requirements.txt
```

### 4. Install franka_sim

Navigate to the `franka_sim` directory and install:

```bash
cd ../franka_sim
pip install -e .
pip install -r requirements.txt
```

### 5. Install serl_robot_infra

Navigate to the `serl_robot_infra` directory and install:

```bash
cd ../serl_robot_infra
pip install -e .
```

### 6. Install demos

Navigate to the `demos` directory and install:

```bash
cd ../demos
pip install -e .
```

Navigation
----------
- [Home](../README.md)
- [Overview](overview.md)
- [Run in simulation](run_sim.md)
- [Run on the real robot](run_realrobot.md)
- [Training options](sim_training.md)
- [Collecting demonstrations](sim_demonstrations.md)
