# Collecting Demonstrations 

The `demos/` folder contains utilities to **collect and record human teleoperated demonstrations** in simulation. 

---

## Recording demos via keyboard teleoperation

Use a keyboard controller to teleoperate the Franka arm in simulation. Each trajectory is automatically recorded.

### Reach task:

```bash
python demos/demos/franka_reach_demo_script.py
```

### Pick-and-place task:

```bash
python demos/demos/franka_pick_n_place_demo_script.py
```
---

## Keyboard controls

Typical keyboard controls for demo collection (check your script for exact bindings):

| Key | Action |
|-----|--------|
| Arrow keys or WASD | Move end-effector (X, Y, Z) |
| Q/E | Rotate gripper |
| Space | Toggle gripper open/close |
| R | Reset environment |
| Esc | Exit |

---

## Loading and inspecting demos

The `demoHandling.py` utilities let you inspect and manipulate demo trajectories:

```python
from demos.demoHandling import load_demos, save_demos

# Load recorded trajectories
trajectories = load_demos('/path/to/demo/file.pkl')

# Inspect structure
print(f"Number of trajectories: {len(trajectories)}")
for i, traj in enumerate(trajectories):
    print(f"Trajectory {i}: {len(traj)} steps")
    # Inspect fields: traj['observations'], traj['actions'], traj['rewards']

# Save processed trajectories
save_demos(trajectories, '/path/to/new/demo/file.pkl')
```

---

## Using demos for training

Once you have collected demonstrations, use them to initialize behavior cloning:

<!-- ### Option 1: Direct BC initialization -->

```bash
cd examples/async_drq_sim
bash run_learner.sh --demo_path /path/to/your/demos.pkl
```

<!-- ### Option 2: RLDS format (recommended)

For better compatibility with other frameworks, convert to RLDS format:

```python
# Convert demos to RLDS (requires oxe_envlogger)
from oxe_envlogger import RLDS_Converter
converter = RLDS_Converter()
converter.convert_trajectory_file('/path/to/demos.pkl', '/path/to/rlds/output')
```

Then load with:
```bash
cd examples/async_drq_sim
bash run_learner.sh --preload_rlds_path /path/to/rlds/output
``` -->

Navigation
----------
- [Home](../README.md)
- [Overview](overview.md)
- [Installation guide](installation.md)
- [Run in simulation](run_sim.md)
- [Training options](sim_training.md)
- [Run on the real robot](run_realrobot.md)
