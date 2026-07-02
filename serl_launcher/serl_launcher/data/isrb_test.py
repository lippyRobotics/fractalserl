"""Standalone test for IndexedSymmetryReplayBuffer. No env/sim dependency.

Run: python isrb_test.py
"""
import numpy as np
import gym
from gym.spaces import Box, Dict as DictSpace

from serl_launcher.data.indexed_symmetry_replay_buffer import IndexedSymmetryReplayBuffer

WORKSPACE_WIDTH = 0.3
STARTING_BRANCH_COUNT = 3
TOTAL_BRANCHES = STARTING_BRANCH_COUNT ** 2
X_OBS_IDX = np.array([4])
Y_OBS_IDX = np.array([5])
STATE_DIM = 7
IMG_H, IMG_W = 16, 16
NUM_STACK = 1


def make_spaces():
    observation_space = DictSpace(
        {
            "state": Box(low=-np.inf, high=np.inf, shape=(NUM_STACK, STATE_DIM), dtype=np.float32),
            "front": Box(low=0, high=255, shape=(NUM_STACK, IMG_H, IMG_W, 3), dtype=np.uint8),
            "wrist_1": Box(low=0, high=255, shape=(NUM_STACK, IMG_H, IMG_W, 3), dtype=np.uint8),
        }
    )
    action_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    return observation_space, action_space


def make_buffer(capacity, front_M=None, world_fixed_img_keys=(), img_keys=("front", "wrist_1")):
    observation_space, action_space = make_spaces()
    buffer = IndexedSymmetryReplayBuffer(
        observation_space=observation_space,
        action_space=action_space,
        capacity=capacity,
        workspace_width=WORKSPACE_WIDTH,
        x_obs_idx=X_OBS_IDX,
        y_obs_idx=Y_OBS_IDX,
        branch_method="constant",
        split_method="never",
        img_keys=list(img_keys),
        kwargs={"starting_branch_count": STARTING_BRANCH_COUNT},
        front_M=front_M,
        world_fixed_img_keys=world_fixed_img_keys,
    )
    buffer.seed(0)
    return buffer


def make_transition(step, done=False):
    state = np.full((NUM_STACK, STATE_DIM), float(step), dtype=np.float32)
    n_state = np.full((NUM_STACK, STATE_DIM), float(step + 1), dtype=np.float32)
    front = np.full((NUM_STACK, IMG_H, IMG_W, 3), step % 256, dtype=np.uint8)
    n_front = np.full((NUM_STACK, IMG_H, IMG_W, 3), (step + 1) % 256, dtype=np.uint8)
    wrist_1 = np.full((NUM_STACK, IMG_H, IMG_W, 3), step % 256, dtype=np.uint8)
    n_wrist_1 = np.full((NUM_STACK, IMG_H, IMG_W, 3), (step + 1) % 256, dtype=np.uint8)

    return dict(
        observations={"state": state, "front": front, "wrist_1": wrist_1},
        next_observations={"state": n_state, "front": n_front, "wrist_1": n_wrist_1},
        actions=np.full((7,), float(step), dtype=np.float32),
        rewards=np.float32(step),
        masks=np.float32(0.0 if done else 1.0),
        dones=bool(done),
    )


def expected_delta(transformation_index):
    x_cell, y_cell = np.divmod(transformation_index, STARTING_BRANCH_COUNT)
    base_diff = -WORKSPACE_WIDTH / 2.0
    dx = (2 * x_cell + 1) * WORKSPACE_WIDTH / (2 * STARTING_BRANCH_COUNT) + base_diff
    dy = (2 * y_cell + 1) * WORKSPACE_WIDTH / (2 * STARTING_BRANCH_COUNT) + base_diff
    return dx, dy


def test_1_raw_storage_and_len():
    buffer = make_buffer(capacity=20)
    n = 6
    for i in range(n):
        buffer.insert(make_transition(i, done=(i == n - 1)))

    assert len(buffer) == n, f"expected len {n}, got {len(buffer)}"

    for i in range(n):
        state_i = buffer.dataset_dict["observations"]["state"][i]
        assert np.allclose(state_i, float(i)), f"row {i}: expected raw state {i}, got {state_i}"

    print("\033[32mTEST PASSED\033[0m test_1_raw_storage_and_len")


def test_2_deterministic_transform():
    buffer = make_buffer(capacity=20)
    n = 6
    for i in range(n):
        buffer.insert(make_transition(i, done=(i == n - 1)))

    i = 2
    tidx = np.array([0, 4, 8], dtype=np.int32)
    batch = buffer.sample(batch_size=3, indx=np.array([i, i, i]), transformation_index=tidx)

    state = np.array(batch["observations"]["state"])
    raw = float(i)

    expected_dx = {0: -0.10, 4: 0.00, 8: 0.10}
    for j, t in enumerate(tidx):
        got_x = state[j, ..., X_OBS_IDX[0]]
        got_y = state[j, ..., Y_OBS_IDX[0]]
        exp = raw + expected_dx[int(t)]
        assert np.allclose(got_x, exp, atol=1e-5), f"t={t}: x expected {exp}, got {got_x}"
        assert np.allclose(got_y, exp, atol=1e-5), f"t={t}: y expected {exp}, got {got_y}"

    # non x/y columns identical across the three samples
    other_idx = [k for k in range(STATE_DIM) if k not in (X_OBS_IDX[0], Y_OBS_IDX[0])]
    for k in other_idx:
        col = state[:, 0, k] if state.ndim == 3 else state[:, k]
        assert np.allclose(col, col[0]), f"column {k} differs across transforms: {col}"

    assert np.allclose(np.array(batch["actions"]), batch["actions"][0])
    assert np.allclose(np.array(batch["rewards"]), batch["rewards"][0])
    assert np.allclose(np.array(batch["masks"]), batch["masks"][0])
    assert np.array_equal(np.array(batch["dones"]), np.full(3, batch["dones"][0]))

    print("\033[32mTEST PASSED\033[0m test_2_deterministic_transform")


def test_3_batch_key():
    buffer = make_buffer(capacity=20)
    n = 6
    for i in range(n):
        buffer.insert(make_transition(i, done=(i == n - 1)))

    batch = buffer.sample(batch_size=5)
    assert "transformation_index" in batch
    tidx = np.array(batch["transformation_index"])
    assert tidx.dtype == np.int32
    assert tidx.shape == (5,)
    assert np.all(tidx >= 0) and np.all(tidx < TOTAL_BRANCHES)

    restricted_keys = ["observations", "actions", "transformation_index"]
    batch2 = buffer.sample(batch_size=5, keys=restricted_keys)
    assert "transformation_index" in batch2

    print("\033[32mTEST PASSED\033[0m test_3_batch_key")


def test_4_obs_next_obs_coupling():
    buffer = make_buffer(capacity=20)
    n = 6
    for i in range(n):
        buffer.insert(make_transition(i, done=(i == n - 1)))

    idx = np.array([1, 3])
    tidx = np.array([2, 7], dtype=np.int32)
    batch = buffer.sample(batch_size=2, indx=idx, transformation_index=tidx)

    obs_state = np.array(batch["observations"]["state"])
    next_state = np.array(batch["next_observations"]["state"])

    raw_obs = idx.astype(np.float32)
    raw_next = (idx + 1).astype(np.float32)

    diff = next_state[:, 0, X_OBS_IDX[0]] - obs_state[:, 0, X_OBS_IDX[0]]
    expected_diff = raw_next - raw_obs
    assert np.allclose(diff, expected_diff), f"transform did not cancel: {diff} vs {expected_diff}"

    print("\033[32mTEST PASSED\033[0m test_4_obs_next_obs_coupling")


def test_5_image_pointer_reconstruction():
    buffer = make_buffer(capacity=20)
    n = 6
    for i in range(n):
        buffer.insert(make_transition(i, done=(i == n - 1)))

    idx = np.array([3])
    batch = buffer.sample(batch_size=1, indx=idx, transformation_index=np.array([4], dtype=np.int32))

    obs_front = np.array(batch["observations"]["front"])
    next_front = np.array(batch["next_observations"]["front"])
    assert obs_front.shape[1] == NUM_STACK
    assert np.all(obs_front == 3), f"expected frame value 3, got unique {np.unique(obs_front)}"
    assert np.all(next_front == 4), f"expected frame value 4, got unique {np.unique(next_front)}"

    batch_packed = buffer.sample(
        batch_size=1, indx=idx, transformation_index=np.array([4], dtype=np.int32), pack_obs_and_next_obs=True
    )
    packed_front = np.array(batch_packed["observations"]["front"])
    assert packed_front.shape[1] == NUM_STACK + 1, f"expected T+1 frames, got shape {packed_front.shape}"

    print("\033[32mTEST PASSED\033[0m test_5_image_pointer_reconstruction")


def test_6_ring_wrap():
    buffer = make_buffer(capacity=5)
    # 2 episodes across 12 transitions, capacity=5 (real transitions)
    step = 0
    for ep in range(2):
        ep_len = 6
        for t in range(ep_len):
            done = t == ep_len - 1
            buffer.insert(make_transition(step, done=done))
            step += 1

    assert len(buffer) == 5, f"expected len 5 (capacity), got {len(buffer)}"

    valid_range = min(step, 5)
    for i in range(valid_range):
        tidx = np.array([4], dtype=np.int32)  # center cell, identity transform
        batch = buffer.sample(batch_size=1, indx=np.array([i]), transformation_index=tidx)
        obs_front = np.array(batch["observations"]["front"])
        next_front = np.array(batch["next_observations"]["front"])
        # frames should be self-consistent: next == obs + 1 (mod 256)
        expected_next = (obs_front.astype(np.int32) + 1) % 256
        assert np.array_equal(next_front.astype(np.int32), expected_next), (
            f"row {i}: ring-wrap image reconstruction incorrect"
        )

    print("\033[32mTEST PASSED\033[0m test_6_ring_wrap")


def test_7_homography():
    scale = IMG_W / WORKSPACE_WIDTH
    front_M = np.array(
        [
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    buffer = make_buffer(
        capacity=10, front_M=front_M, world_fixed_img_keys=("front",), img_keys=("front", "wrist_1")
    )

    # Insert a transition with a bright 2x2 marker centered in the front image.
    step = 0
    obs_front = np.zeros((NUM_STACK, IMG_H, IMG_W, 3), dtype=np.uint8)
    cy, cx = IMG_H // 2, IMG_W // 2
    obs_front[:, cy : cy + 2, cx : cx + 2, :] = 255
    wrist_1 = np.full((NUM_STACK, IMG_H, IMG_W, 3), 7, dtype=np.uint8)

    transition = dict(
        observations={"state": np.zeros((NUM_STACK, STATE_DIM), np.float32), "front": obs_front, "wrist_1": wrist_1},
        next_observations={
            "state": np.zeros((NUM_STACK, STATE_DIM), np.float32),
            "front": obs_front,
            "wrist_1": wrist_1,
        },
        actions=np.zeros((7,), np.float32),
        rewards=np.float32(0.0),
        masks=np.float32(1.0),
        dones=True,
    )
    buffer.insert(transition)

    # Center cell (identity transform) -> warped ~ original.
    batch_center = buffer.sample(
        batch_size=1, indx=np.array([0]), transformation_index=np.array([4], dtype=np.int32)
    )
    warped_center = np.array(batch_center["observations"]["front"])[0, 0].astype(np.float32)
    orig = obs_front[0].astype(np.float32)
    assert np.allclose(warped_center, orig, atol=40.0), "identity homography warp deviates too much from original"

    # transformation_index=8 -> delta (+0.10, +0.10) -> marker displaced by ~scale*0.10 px
    batch_shift = buffer.sample(
        batch_size=1, indx=np.array([0]), transformation_index=np.array([8], dtype=np.int32)
    )
    warped_shift = np.array(batch_shift["observations"]["front"])[0, 0].astype(np.float32)

    # Find brightest pixel location in each warped image (marker center).
    def marker_center(img):
        gray = img.mean(axis=-1)
        ys, xs = np.where(gray > gray.max() - 1.0)
        return ys.mean(), xs.mean()

    cy0, cx0 = marker_center(warped_center)
    cy1, cx1 = marker_center(warped_shift)

    expected_shift_px = scale * 0.10  # ~5.3 px per axis (transformation_index=8 shifts both x and y)
    dy = cy1 - cy0
    dx = cx1 - cx0
    assert abs(dx - expected_shift_px) < 2.0, (
        f"expected x marker shift ~{expected_shift_px:.2f}px, got {dx:.2f}px"
    )
    assert abs(dy - expected_shift_px) < 2.0, (
        f"expected y marker shift ~{expected_shift_px:.2f}px, got {dy:.2f}px"
    )

    # wrist_1 is never warped -> bit-identical
    wrist_center = np.array(batch_center["observations"]["wrist_1"])
    wrist_shift = np.array(batch_shift["observations"]["wrist_1"])
    assert np.array_equal(wrist_center, wrist_shift)
    assert np.all(wrist_center == 7)

    print("\033[32mTEST PASSED\033[0m test_7_homography")


def test_8_determinism():
    buffer_a = make_buffer(capacity=20)
    buffer_b = make_buffer(capacity=20)

    n = 6
    for i in range(n):
        t = make_transition(i, done=(i == n - 1))
        buffer_a.insert(t)
        buffer_b.insert(make_transition(i, done=(i == n - 1)))

    batch_a = buffer_a.sample(batch_size=4)
    batch_b = buffer_b.sample(batch_size=4)

    assert np.array_equal(np.array(batch_a["transformation_index"]), np.array(batch_b["transformation_index"]))
    assert np.allclose(np.array(batch_a["observations"]["state"]), np.array(batch_b["observations"]["state"]))
    assert np.array_equal(np.array(batch_a["observations"]["front"]), np.array(batch_b["observations"]["front"]))

    print("\033[32mTEST PASSED\033[0m test_8_determinism")


def test_9_downstream_compatibility():
    import jax
    from serl_launcher.utils.train_utils import concat_batches

    buffer = make_buffer(capacity=20)
    n = 6
    for i in range(n):
        buffer.insert(make_transition(i, done=(i == n - 1)))

    batch = buffer.sample(batch_size=4)
    combined = concat_batches(batch, batch, axis=0)
    assert "transformation_index" in combined
    assert combined["transformation_index"].shape[0] == 8

    device_batch = jax.device_put(combined)
    assert "transformation_index" in device_batch

    print("\033[32mTEST PASSED\033[0m test_9_downstream_compatibility")


def test_10_config_guard():
    observation_space, action_space = make_spaces()
    try:
        IndexedSymmetryReplayBuffer(
            observation_space=observation_space,
            action_space=action_space,
            capacity=10,
            workspace_width=WORKSPACE_WIDTH,
            x_obs_idx=X_OBS_IDX,
            y_obs_idx=Y_OBS_IDX,
            branch_method="fractal",
            split_method="never",
            img_keys=["front", "wrist_1"],
            kwargs={"starting_branch_count": STARTING_BRANCH_COUNT},
        )
        raise RuntimeError("expected AssertionError for branch_method='fractal'")
    except AssertionError:
        pass

    print("\033[32mTEST PASSED\033[0m test_10_config_guard")


def main():
    test_1_raw_storage_and_len()
    test_2_deterministic_transform()
    test_3_batch_key()
    test_4_obs_next_obs_coupling()
    test_5_image_pointer_reconstruction()
    test_6_ring_wrap()
    test_7_homography()
    test_8_determinism()
    test_9_downstream_compatibility()
    test_10_config_guard()
    print("\nfinished!\n")


if __name__ == "__main__":
    main()
