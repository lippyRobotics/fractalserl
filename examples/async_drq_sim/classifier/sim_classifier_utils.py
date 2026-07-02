import os
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import gym
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

from serl_launcher.networks.reward_classifier import create_classifier
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper


def configure_jax_cpu_only():
    """Avoid eager CUDA plugin loading when JAX is explicitly CPU-only."""
    platforms = os.environ.get("JAX_PLATFORMS", "")
    if [item.strip().lower() for item in platforms.split(",") if item.strip()] != [
        "cpu"
    ]:
        return

    try:
        import jax_plugins.xla_cuda12 as cuda_plugin
    except ImportError:
        return

    # Some JAX CUDA plugin versions eagerly dlopen CUDA libraries during plugin
    # discovery even when JAX_PLATFORMS=cpu. Discovery still runs, but this
    # prevents the unused CUDA initializer from crashing a CPU-only process.
    cuda_plugin.initialize = lambda: None


def infer_image_keys(observation_space: gym.spaces.Dict) -> List[str]:
    image_keys = [
        key
        for key, space in observation_space.spaces.items()
        if key != "state" and len(getattr(space, "shape", ())) >= 3
    ]
    if not image_keys:
        raise ValueError(
            "No image keys found in observation space. Expected SERL-formatted "
            "observations with keys such as 'front' and 'wrist'."
        )
    return image_keys


def validate_image_keys(observation_space: gym.spaces.Dict, image_keys: Sequence[str]):
    missing = [key for key in image_keys if key not in observation_space.spaces]
    if missing:
        raise ValueError(
            f"Invalid image keys {missing}. Available keys: "
            f"{list(observation_space.spaces.keys())}"
        )


def make_serl_vision_env(env_name: str, render: bool = False):
    env = gym.make(
        env_name,
        render_mode="human" if render else "rgb_array",
        disable_env_checker=True,
    )
    if not isinstance(env.observation_space, gym.spaces.Dict):
        raise ValueError(f"{env_name} does not expose Dict observations.")
    if "images" in env.observation_space.spaces:
        env = SERLObsWrapper(env)
    elif "state" not in env.observation_space.spaces:
        raise ValueError(
            f"{env_name} observations are not compatible with SERLObsWrapper."
        )
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    return env


def select_classifier_obs(obs, image_keys: Iterable[str], use_proprio: bool):
    keys = list(image_keys)
    if use_proprio:
        keys.append("state")
    return {key: obs[key] for key in keys}


def load_classifier_func(
    ckpt_path: str,
    sample_obs,
    image_keys,
    checkpoint_step=None,
    threshold=0.5,
    use_proprio=False,
) -> Callable:
    del threshold
    if not ckpt_path:
        raise ValueError("A classifier checkpoint path is required.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Classifier checkpoint path not found: {ckpt_path}")

    key = jax.random.PRNGKey(0)
    classifier = create_classifier(
        key,
        select_classifier_obs(sample_obs, image_keys, use_proprio),
        list(image_keys),
        use_proprio=use_proprio,
    )
    classifier = checkpoints.restore_checkpoint(
        ckpt_path,
        target=classifier,
        step=checkpoint_step,
    )
    if classifier is None:
        raise ValueError(f"Unable to restore classifier checkpoint from {ckpt_path}")

    @jax.jit
    def _predict(obs):
        obs = select_classifier_obs(obs, image_keys, use_proprio)
        return classifier.apply_fn({"params": classifier.params}, obs, train=False)

    return _predict


def classifier_probability(classifier_fn: Callable, obs) -> tuple[float, float]:
    logit = classifier_fn(jax.device_put(obs))
    logit = float(np.asarray(jax.device_get(logit)).reshape(-1)[0])
    prob = float(jax.nn.sigmoid(jnp.asarray(logit)))
    return logit, prob


class ClassifierRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        classifier_fn: Callable,
        threshold: float = 0.5,
        terminate_on_success: bool = True,
        zero_env_reward: bool = False,
    ):
        super().__init__(env)
        self.classifier_fn = classifier_fn
        self.threshold = threshold
        self.terminate_on_success = terminate_on_success
        self.zero_env_reward = zero_env_reward

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        logit, prob = classifier_probability(self.classifier_fn, obs)
        success = prob >= self.threshold
        classifier_reward = 1.0 if success else 0.0
        reward = classifier_reward if self.zero_env_reward else reward + classifier_reward
        done = bool(done or (success and self.terminate_on_success))
        info = dict(info)
        info["classifier_logit"] = logit
        info["classifier_prob"] = prob
        info["classifier_success"] = bool(success)
        return obs, reward, done, truncated, info


def get_unwrapped(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def get_sensor(env, name: str) -> Optional[np.ndarray]:
    base = get_unwrapped(env)
    try:
        return np.asarray(base._data.sensor(name).data, dtype=np.float32).copy()
    except Exception:
        return None


def pickcube_success(env, min_lift: float = 0.12) -> bool:
    base = get_unwrapped(env)
    block_pos = get_sensor(env, "block_pos")
    if block_pos is None:
        return False
    z_init = getattr(base, "_z_init", None)
    if z_init is None:
        return False
    return bool(block_pos[2] >= float(z_init) + min_lift)


def force_pickcube_success_state(env, lift: float = 0.22) -> bool:
    base = get_unwrapped(env)
    try:
        import mujoco

        qpos = base._data.jnt("block").qpos
        qpos[2] = float(getattr(base, "_z_init", qpos[2])) + lift
        base._data.jnt("block").qvel[:] = 0.0
        mujoco.mj_forward(base._model, base._data)
        return True
    except Exception:
        return False


def scripted_pickcube_action(env, step: int, max_steps: int) -> np.ndarray:
    tcp = get_sensor(env, "2f85/pinch_pos")
    block = get_sensor(env, "block_pos")
    if tcp is None or block is None:
        return env.action_space.sample()

    base = get_unwrapped(env)
    if step == 0 or not hasattr(base, "_serl_scripted_pick_state"):
        base._serl_scripted_pick_state = {
            "stage": "approach",
            "stage_steps": 0,
            "stable_steps": 0,
        }
    controller = base._serl_scripted_pick_state

    approach_limit = max(40, int(max_steps * 0.35))
    descend_limit = max(35, int(max_steps * 0.32))
    close_limit = max(28, int(max_steps * 0.15))

    def track(target, max_action):
        return np.clip((target - tcp) / 0.1, -max_action, max_action)

    def update_stability(target, tolerance):
        if np.linalg.norm(target - tcp) <= tolerance:
            controller["stable_steps"] += 1
        else:
            controller["stable_steps"] = 0

    def advance(stage):
        controller["stage"] = stage
        controller["stage_steps"] = 0
        controller["stable_steps"] = 0

    stage = controller["stage"]
    if stage == "approach":
        target = block + np.asarray([0.0, 0.0, 0.14], dtype=np.float32)
        update_stability(target, tolerance=0.025)
        delta = track(target, max_action=0.15)
        gripper = -0.15
        if controller["stable_steps"] >= 8 or controller["stage_steps"] >= approach_limit:
            advance("descend")
    elif stage == "descend":
        target = block + np.asarray([0.0, 0.0, 0.005], dtype=np.float32)
        update_stability(target, tolerance=0.012)
        delta = track(target, max_action=0.06)
        gripper = -0.10
        if controller["stable_steps"] >= 10 or controller["stage_steps"] >= descend_limit:
            advance("close")
    elif stage == "close":
        target = block + np.asarray([0.0, 0.0, 0.005], dtype=np.float32)
        delta = track(target, max_action=0.10)
        gripper = 0.06
        if controller["stage_steps"] >= close_limit:
            advance("lift")
    else:
        z_init = float(getattr(base, "_z_init", block[2]))
        target = np.asarray(
            [block[0], block[1], z_init + 0.22], dtype=np.float32
        )
        delta = track(target, max_action=0.15)
        gripper = 0.05

    controller["stage_steps"] += 1
    return np.asarray([delta[0], delta[1], delta[2], gripper], dtype=np.float32)


def random_action(env, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)


def ensure_parent(path: str):
    parent = Path(path).expanduser().parent
    parent.mkdir(parents=True, exist_ok=True)
