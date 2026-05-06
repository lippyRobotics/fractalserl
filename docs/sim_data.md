# Saving and Loading Trajectories
<!-- 
FractalSERL supports the standard **RLDS (Robot Learning Data Standard)** format for saving and loading trajectories. This enables compatibility with other robot-learning frameworks (e.g., Robotics Transformer X, RT-1).

---

## Overview

**RLDS** is a format for storing robot learning datasets in a standardized way:
- **TFRecord format** — Efficient storage and streaming
- **Modular structure** — Separate metadata, episode info, and trajectory shards
- **Language-agnostic** — Can be used with any learning framework
- **Interoperable** — Share datasets across teams and projects

---

## Saving trajectories from replay buffer

During training, capture environment transitions and save them in RLDS format.

### Basic usage:

```bash
cd examples/async_drq_sim
bash run_learner.sh --log_rlds_path /path/to/save
```

### Output structure:

The learner creates a structured RLDS dataset:

```
/path/to/save/
├── dataset_info.json          # Metadata (feature schema, episode counts)
├── features.json              # Feature definitions
├── serl_rlds_dataset-train.tfrecord-00000
├── serl_rlds_dataset-train.tfrecord-00001
├── serl_rlds_dataset-train.tfrecord-00002
└── ... (more shards for large datasets)
```

Each TFRecord file is a shard of the full dataset, enabling parallel loading and training.

---

## Understanding the RLDS output

### `dataset_info.json`

Contains high-level information about the dataset:

```json
{
  "name": "franka_sim_drq_training",
  "license": "CC-BY-4.0",
  "splits": {
    "train": 1000
  },
  "episodes": 50,
  "steps": 50000,
  "example": {
    ...
  }
}
```

### `features.json`

Defines the structure of each step in the dataset:

```json
{
  "observation": "uint8",
  "action": "float32",
  "reward": "float32",
  "is_terminal": "bool",
  "is_first": "bool"
}
```

---

## Loading pre-saved trajectories

Use previously saved RLDS data to resume training or initialize with demonstrations.

### Method 1: Behavior cloning with RLDS

```bash
cd examples/async_drq_sim
bash run_learner.sh --preload_rlds_path /path/to/rlds/data
```

The learner initializes the policy using behavior cloning, then switches to RL fine-tuning.

### Method 2: Replay buffer initialization

```bash
cd examples/async_drq_sim
bash run_learner.sh --preload_rlds_path /path/to/rlds/data --bc_steps 5000
```

Adjust `--bc_steps` to control the behavior-cloning phase duration.

---

## Working with RLDS programmatically

### Loading RLDS data in Python:

```python
import tensorflow_datasets as tfds

# Load RLDS dataset
dataset = tfds.load('path/to/rlds/data', split='train', shuffle_files=True)

# Iterate through trajectories
for episode in dataset:
    observations = episode['observation']
    actions = episode['action']
    rewards = episode['reward']
    
    print(f"Episode length: {len(observations)}")
```

### Converting custom demos to RLDS:

If you have pickled demos (from `demos/`), convert them to RLDS:

```python
from oxe_envlogger import RLDS_Converter

converter = RLDS_Converter()
converter.convert_trajectory_file(
    input_path='demos_franka_reach.pkl',
    output_dir='/path/to/rlds/output'
)
```

---

## Saving and loading during training

### Periodic checkpoint saving:

```bash
cd examples/async_drq_sim
bash run_learner.sh --log_rlds_path /data/checkpoints --log_interval 10000
```

Every 10,000 steps, a new RLDS checkpoint is saved.

### Resuming from checkpoint:

```bash
cd examples/async_drq_sim
bash run_learner.sh --preload_rlds_path /data/checkpoints/step_50000
```

---

## Best practices

1. **Use RLDS for long-term storage** — More efficient and interoperable than pickle files
2. **Organize by task/experiment** — Use descriptive directory names:
   ```
   /data/
   ├── reach_sac_state/
   ├── reach_drq_image/
   ├── pick_drq_image_bc/
   └── pick_drq_multimodal/
   ```
3. **Document your dataset** — Add a `README.md` in each dataset directory
4. **Compute dataset statistics** — Save mean/std for normalization
5. **Version your data** — Include timestamps or git hashes in checkpoint names

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **TFRecord not found** | Ensure `--log_rlds_path` points to an existing directory with write permissions. |
| **Out of memory when loading** | Use `shuffle_files=False` and load in batches. |
| **Incompatible feature schema** | Ensure RLDS features match your environment's observation/action space. |
| **Slow TFRecord reading** | Use multiple worker processes: `num_parallel_reads=4` in `tf.data.TFRecordDataset()`. |

---

## Next steps

- **Using RLDS for initialization?** See [Training options](sim_training.md) → Option 3 (DRQ+BC).
- **Recording custom demos?** See [Collecting demonstrations](sim_demonstrations.md).
- **Advanced training setup?** See [Advanced training](sim_advanced.md).
- **Back to main guide?** See [run_sim.md](run_sim.md). -->
