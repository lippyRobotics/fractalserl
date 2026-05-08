# Overview — FractalSERL

This document summarizes the code structure, system design, folder layout, and next steps for FractalSERL — our contribution and extension to the original SERL project.

Module summary
------------------------

| Module | Purpose | Key files / subfolders |
|---|---|---|
| `demos/` | Demo scripts and logging | `demos/`, `oxe_envlogger/` |
| `examples/` | Training pipelines and launch scripts | `async_*/`, `bc_policy.py`, `run_*.sh` |
| `franka_sim/` | Simulation environments and controllers | `envs/`, `controllers/`, `mujoco_gym_env.py` |
| `serl_launcher/` | Core RL algorithms, agents, and utilities | `agents/`, `networks/`, `data/`, `wrappers/`, `utils/` |
| `serl_robot_infra/` | Real-robot interfaces and servers | `franka_env/`, `robot_servers/`, `camera/`, `spacemouse/` |

Each top-level package includes a `setup.py` so it can be installed in editable mode (`pip install -e .`) during development. A `requirements.txt` file lists runtime dependencies used by demo and example scripts.


System design
-----------------------

![Runtime architecture: actor/learner architecture](images/software_design.png)

Actor node collects data from Gym-compatible environments (sim or real) and push transitions to a datastore/replay buffer; learner node consumes that data to update policies and periodically push updated weights back to actor node. Communication is asynchronous, using `agentlace` in experiments, so collection and learning scale independently.

Key points:
- **Actor node:** environment stepping, action sampling, transition senders.
- **Learner node:** gradient updates, replay-buffer consumer, policy synchronization.
- **Environment wrappers:** `serl_launcher/wrappers/` provide a consistent Gym API across sim and real.
- **Hardware servers:** ROS code and hardware commands are located in `serl_robot_infra/robot_servers/`.

Navigation 
-----------------------
- [Home](../README.md)
- [Installation guide](installation.md)
- [Run in simulation](run_sim.md)
- [Run on the real robot](run_realrobot.md)
- [Training options](sim_training.md)
- [Collecting demonstrations](sim_demonstrations.md)


