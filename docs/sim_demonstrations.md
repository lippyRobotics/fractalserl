# Collecting Demonstrations 
<!-- 
The `demos/` folder contains utilities to **collect and record human teleoperated demonstrations** in simulation. These recordings can be:
- Used as initialization data for behavior cloning (See [Training options](sim_training.md) → Option 3)
- Saved and replayed for inspection
- Logged in standard RLDS format for compatibility with other learning frameworks

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

### Demo script behavior:

- **Recording:** State, image, and action transitions are automatically recorded
- **Saving:** Trajectories are saved to a default location (configurable in script)
- **Key bindings:** Depend on the specific script (see script comments for details)
- **Episode length:** Limited to 100 steps per episode

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

### Option 1: Direct BC initialization

```bash
cd examples/async_drq_sim
bash run_learner.sh --demo_path /path/to/your/demos.pkl
```

### Option 2: RLDS format (recommended)

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
```

---

## Demo structure (what gets recorded)

Each demo trajectory contains:

```
{
  'observations': array of shape (T, obs_dim),  # State or image observations
  'actions': array of shape (T, 3),              # 3D delta end-effector movements
  'rewards': array of shape (T,),                # Task rewards
  'terminals': array of shape (T,),              # Episode termination flags
  'truncations': array of shape (T,),            # Time limit flags (100 steps)
}
```

Where `T` is the trajectory length (typically ≤ 100 steps).

---

## Next steps

- **Using demos for training?** See [Training options](sim_training.md) → Option 3 (DRQ+BC).
- **Saving to standard format?** See [Saving and loading data](sim_data.md).
- **Troubleshooting?** See [Advanced training](sim_advanced.md).
- **Back to main guide?** See [run_sim.md](run_sim.md). -->
