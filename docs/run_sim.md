# Running FractalSERL in Simulation 

This guide covers how to train RL policies in MuJoCo simulation using FractalSERL. The simulation environment includes a Franka Panda robot arm and manipulation tasks (e.g., reaching, cube lift). We support both state-based and image-based training, with optional behavior-cloning initialization via demonstrations.

## Prerequisites

Before starting, ensure you have:

1. **Installed FractalSERL** — follow the [installation guide](installation.md) to set up all packages.
2. **Verified `franka_sim`** — test the simulation environment:
   ```bash
   python franka_sim/franka_sim/test/test_gym_env_human.py
   ```
3. **(Optional) `tmux` installed** — for convenient parallel actor/learner launch:
   ```bash
   sudo apt install tmux
   ```


## Environment overview

The default simulation task is **Reach**, where:

- **State space:** end-effector position/orientation, velocities, gripper state, target block position.
- **Action space:** 3D delta movements (Δx, Δy, Δz).
- **Reward:** Dense, based on distance to target: $r = \text{clip}(e^{-20d}, 0, 1)$ where $d$ is Euclidean distance.
- **Images:** When enabled, two RGB wrist-mounted camera views replace explicit block position.
- **Episode length:** 100 steps.

This task enables testing of **branched symmetries and fractal variants** as described in the paper (pure translations applied to positional components only).

## Training guide

Choose your training approach based on your needs:

- **[Training options](sim_training.md)** — State-based SAC, Image-based DRQ, or DRQ with behavior cloning initialization.
- **[Collecting demonstrations](sim_demonstrations.md)** — Record human teleoperated demos for initialization or analysis.

Navigation
----------
- [Home](../README.md)
- [Overview](overview.md)
- [Installation guide](installation.md)
- [Training options](sim_training.md)
- [Collecting demonstrations](sim_demonstrations.md)
- [Run on the real robot](run_realrobot.md)
