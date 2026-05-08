# Training Options 


Choose your training approach based on your requirements:

- **Option 1: State-based SAC** — Simplest and fastest. No images needed.
- **Option 2: Image-based DRQ** — Vision-based policy learning with pre-trained ResNet encoder.
- **Option 3: Image-based DRQ + Behavior Cloning** — Pre-train on human demonstrations, then fine-tune with RL.

---

## Option 1: State-based SAC (simplest, fastest)

Train using state observations (no images, less computation).

### Quick start (with `tmux`):
```bash
bash examples/async_sac_state_sim/tmux_launch.sh
```

### Manual setup (two terminals):

**Terminal 1 — Learner:**
```bash
cd examples/async_sac_state_sim
bash run_learner.sh
```

**Terminal 2 — Actor:**
```bash
cd examples/async_sac_state_sim
bash run_actor.sh
```



---

## Option 2: Image-based DRQ (vision-based policy)

Train using visual observations from camera(s). DRQ (Data-Regularized Q-learning) is designed for image-based control.

### Prerequisites:

Download the pre-trained ResNet-10 encoder (required for feature extraction):
```bash
cd examples/async_drq_sim
wget https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl
```

### Quick start (with `tmux`):
```bash
bash examples/async_drq_sim/tmux_launch.sh
```

### Manual setup (two terminals):

**Terminal 1 — Learner:**
```bash
cd examples/async_drq_sim
bash run_learner.sh
```

**Terminal 2 — Actor:**
```bash
cd examples/async_drq_sim
bash run_actor.sh
```
---

## Option 3: Image-based DRQ + Behavior Cloning (with demonstrations)

Pre-train on ~20 human demonstrations, then fine-tune with RL. This combines the best of both worlds:
- **BC phase:** Learn from expert demonstrations
- **RL phase:** Refine policy with environment interaction

### Prerequisites:

Download ResNet encoder and demo trajectories:
```bash
cd examples/async_drq_sim
wget https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl
wget https://github.com/rail-berkeley/serl/releases/download/franka_sim_lift_cube_demos/franka_lift_cube_image_20_trajs.pkl
```

### Quick start (with `tmux`):
```bash
bash examples/async_drq_sim/tmux_rlpd_launch.sh
```

### Manual setup (two terminals):

**Terminal 1 — Learner (with demos):**
```bash
cd examples/async_drq_sim
bash run_learner.sh --demo_path franka_lift_cube_image_20_trajs.pkl
```

**Terminal 2 — Actor:**
```bash
cd examples/async_drq_sim
bash run_actor.sh
```

### Custom demonstrations:

Don't have pre-recorded demos? Create your own:
- See [Collecting demonstrations](sim_demonstrations.md) for keyboard teleoperation
- Save demos and pass with `--demo_path <path>` to the learner

Navigation
----------
- [Home](../README.md)
- [Overview](overview.md)
- [Installation guide](installation.md)
- [Run in simulation](run_sim.md)
- [Collecting demonstrations](sim_demonstrations.md)
- [Run on the real robot](run_realrobot.md)

