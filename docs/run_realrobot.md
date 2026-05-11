# Run with Franka Arm on Real Robot

This guide covers the real-robot setup for FractalSERL. The robot stack is split into a Flask server that sends commands to the robot via ROS and a Gym environment client that communicates with that server over HTTP.

![](./images/robot_infra_interfaces.png)

## Installation for `serl_franka_controllers`

Follow the [SERL Robot Infra README](../serl_robot_infra/README.md) for installation and basic robot operation instructions. That page includes the impedance-based [serl_franka_controllers](https://github.com/rail-berkeley/serl_franka_controllers) setup.

After installation, you should be able to start the robot server and interact with the hardware Gym environment.

> NOTE: The example code below is a template. It assumes you have your own robot setup, camera calibration, data, and checkpoints.

## 1. Peg Insertion


![peg-insert](../docs/images/peg-insert-realrobot.png)

**Example:** [examples/async_peg_insert_drq/](../examples/async_peg_insert_drq/)

**Environment and Default Configuration:** [serl_robot_infra/franka_env/envs/peg_env/](../serl_robot_infra/franka_env/envs/peg_env/)

**Wrapper:** `franka_env.envs.wrappers.SpacemouseIntervention`

Peg insertion is recommended as the first task for validating the real-robot stack. It is the simplest setup for checking the server, cameras, reward target, and training loop.

### Procedure
1. Prepare the peg, board, and workspace. Fix the board in place and mount the peg in the gripper.
2. Mount the wrist cameras and update the camera serial numbers in `peg_env/config.py`.
3. Adjust the wrist-camera mass in Desk so the controller matches the payload.
4. Unlock the robot, enable FCI, and start the server:

	```bash
	python serl_robot_infra/robot_servers/franka_server.py \
		 --gripper_type=<Robotiq|Franka|None> \
		 --robot_ip=<robot_IP> \
		 --gripper_ip=<[Optional] Robotiq_gripper_IP>
	```

5. Use the pose and gripper endpoints to measure the target pose, then update `TARGET_POSE` in `peg_env/config.py`.
6. Set `RANDOM_RESET=False` while debugging the base task.
7. Record demonstrations with the spacemouse:

	```bash
	cd examples/async_peg_insert_drq
	python record_demo.py
	```

8. Update `demo_path` and `checkpoint_path` in the actor and learner scripts, then launch training.
9. Evaluate checkpoints with `--eval_checkpoint_step` and `--eval_n_trajs` in `run_actor.sh`.

### Wrapper stack

```python
env = gym.make('FrankaPegInsert-Vision-v0')
env = GripperCloseEnv(env)
env = SpacemouseIntervention(env)
env = RelativeFrame(env)
env = Quat2EulerWrapper(env)
env = SERLObsWrapper(env)
env = ChunkingWrapper(env)
env = RecordEpisodeStatistics(env)
```

## 2. Cable Routing

![cable-routing](../docs/images/cable-routing-realrobot.png)


**Example:** [examples/async_cable_routing_drq/](../examples/async_cable_routing_drq/)

**Env and default config:** [serl_robot_infra/franka_env/envs/cable_env/](../serl_robot_infra/franka_env/envs/cable_env/) 

Cable routing uses an image-based reward classifier instead of a fixed target pose. Train the classifier on successful and failed trajectories, then pass its checkpoint to the actor and learner scripts.

```bash
python train_reward_classifier.py \
	 --classifier_ckpt_path CHECKPOINT_OUTPUT_DIR \
	 --positive_demo_paths PATH_TO_POSITIVE_DEMO1.pkl \
	 --positive_demo_paths PATH_TO_POSITIVE_DEMO2.pkl \
	 --negative_demo_paths PATH_TO_NEGATIVE_DEMO1.pkl
```

The classifier is used with `franka_env.envs.wrapper.BinaryRewardClassifier` so the policy can train from an observation-based reward.

## 3. Object Relocation TODO

<!-- > Example: [examples/async_bin_relocation_fwbw_drq/](../examples/async_bin_relocation_fwbw_drq/)

> Env and default config: `serl_robot_infra/franka_env/envs/bin_env/`

Object relocation uses forward and backward policies so the robot can move an object between bins and reset itself during training.

### Workflow
1. Record forward and backward trajectories separately.
2. Train a reward classifier for each direction.
3. Launch the actor and both learners:

	```bash
	bash run_actor.sh
	bash run_fw_learner.sh
	bash run_bw_learner.sh
	``` -->

## Navigation

- [Home](../README.md)
- [Overview](overview.md)
- [Installation guide](installation.md)
- [Run in simulation](run_sim.md)
- [Quick start](sim_quick_start.md)
- [Training options](sim_training.md)
- [Collecting demonstrations](sim_demonstrations.md)
