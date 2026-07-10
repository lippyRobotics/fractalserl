import copy
from typing import Iterable, Optional

import cv2
import gym
import numpy as np
from serl_launcher.data.dataset import DatasetDict, _sample
from serl_launcher.data.replay_buffer import ReplayBuffer
from flax.core import frozen_dict

class FractalSymmetryReplayBuffer(ReplayBuffer):
    """Replay buffer that augments each transition with translated copies.

    This class is a regular :class:`ReplayBuffer` with spatial symmetry
    augmentation. A *transition* is one reinforcement-learning step containing
    an observation, the action taken, the reward, the next observation, and
    termination values. Instead of storing that transition only once, this
    buffer places copies on a square grid covering the workspace. The copies
    describe the same relative motion at different absolute ``x`` and ``y``
    positions, so an agent can learn that the behavior transfers across the
    workspace.

    Term

    * A *branch* represents a transformation of the original rollout in a single axis.
       If the current branch count is ``n``, the symmetric grid has ``n`` columns 
      and ``n`` rows, producing``n ** 2`` transformed transitions.
    * *depth* is the depth of the tree with nodes and edges. Branches may change 
      across depths. 
    * A *split* is the event that advances the depth. ``split_method`` controls
      when this happens; ``branch_method`` controls the new branch count.
    * A *transform delta* is the precomputed translations added to the selected
      coordinates. ``x_obs_idx`` and ``y_obs_idx`` identify those coordinates
      in the state vector.
    * An *image stack* is a sequence of consecutive camera frames. Images are
      stored once in a separate circular buffer because every translated state
      copy uses the same pixels.
    * A *front map* is a precomputed dense destination-to-source pixel map for
      one spatial transformation. For each world-fixed camera,
      ``front_maps[key]`` has shape ``(n ** 2, height, width, 2)``. The final
      dimension stores the source ``(pixel_x, pixel_y)`` coordinates consumed by
      :func:`cv2.remap`.
    * A *transformation index* identifies which state translation produced a
      replay entry. It selects the matching front map during sampling and
      remains internal to the replay buffer.

    Construction proceeds as follows:

    1. The branching and splitting methods are selected and their settings are
       read from ``kwargs``.
    2. The largest expected number of transformed copies is calculated so the
       underlying replay buffer can allocate enough space.
    3. If ``img_keys`` is non-empty, compact integer frame pointers replace
       images in the main buffer, while the actual frames are allocated in a
       separate circular buffer.
    4. The parent replay buffer is allocated with
       ``capacity * expected_branches`` entries.
    5. Translation deltas for the initial grid are generated once and reused.

    Inserting one transition then follows this sequence:

    1. The split rule decides whether the trajectory advances to another depth.
    2. The branch rule determines the grid size at that depth. Translation
       deltas are regenerated only when this size changes.
    3. The selected ``x`` and ``y`` state values are moved to a workspace origin
       at ``-workspace_width / 2`` and copied ``current_branch_count ** 2``
       times.
    4. A different grid-cell-center translation is added to every state copy.
       Actions, rewards, masks, and done flags are duplicated without changes.
    5. Camera frames, when present, are stored once and referenced by every
       transformed copy.
    6. The batch of transformed transitions is inserted into the parent buffer.
       At episode end, depth and timestep return to zero.

    Sampling without images behaves like :class:`ReplayBuffer`. With images,
    state data, frame pointers, and transformation indices are sampled first.
    Consecutive frames are reconstructed as observation and next-observation
    stacks. For each sampled world-fixed image, its transformation index selects
    one dense map from ``front_maps`` and :func:`cv2.remap` produces pixels
    consistent with the transformed state. Robot-mounted images bypass this
    remapping, and transformation indices are not returned to the learner.

    Args:
        observation_space: Gym space describing observations. When images are
            used, this must be a dictionary space containing ``"state"`` and
            every key listed in ``img_keys``.
        action_space: Gym space describing actions.
        capacity: Number of original, unaugmented transitions to budget for.
            Internal state capacity is multiplied by the maximum expected
            number of transformed copies.
        workspace_width: Width of the square workspace in state-coordinate
            units.
        x_obs_idx: One-dimensional NumPy array of state-vector positions that
            contain x coordinates.
        y_obs_idx: One-dimensional NumPy array of state-vector positions that
            contain y coordinates.
        branch_method: Rule for choosing grid divisions. ``"fractal"`` grows
            as ``branching_factor ** current_depth``; ``"contraction"`` shrinks
            from a dense grid; ``"disassociated"`` interpolates between minimum
            and maximum counts; and ``"constant"`` keeps one configured count.
        split_method: Rule for advancing depth. ``"time"`` divides an estimated
            trajectory into depth intervals, ``"constant"`` advances on every
            insertion, and ``"never"`` does not advance.
        img_keys: Observation-dictionary keys containing stacked camera images.
            Use an empty list for state-only observations.
        front_M: Calibrated 3x3 homography mapping workspace-plane coordinates
            to pixels in the world-fixed camera image.
        world_fixed_img_keys: Image keys whose frames must be remapped to match
            the sampled state transformation.
        kwargs: Settings required by the selected methods. Depending on the
            configuration these include ``max_depth``, ``branching_factor``,
            ``min_branch_count``, ``max_branch_count``,
            ``disassociated_type``, ``starting_branch_count``,
            ``max_traj_length``, and ``alpha``. For time splitting, ``alpha`` is
            the exponential-moving-average weight used to update the estimated
            trajectory length after each episode.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        workspace_width: int,
        x_obs_idx : np.ndarray,     # x-axis entries to be transformed. Could be object + end-effector or only end-effector
        y_obs_idx : np.ndarray,     # y-axis entries to be transformed. Could be object + end-effector or only end-effector
        branch_method: str,
        split_method: str,
        img_keys: list,
        kwargs: dict,
        front_M: Optional[np.ndarray] = None,
        world_fixed_img_keys: Iterable[str] = (),
    ):

        # Initialize values
        self.debug_time = True
        self.current_branch_count = 1           # number of branches currently. can change with time unless constant.
        self.update_max_traj_length = False
        self.workspace_width = workspace_width
        self.img_keys = img_keys
        self._img_insert_index_ = 0
        self._num_original_transitions_inserted = 0

        # Homography configuration for world-fixed cameras. front_M is the
        # calibrated 3x3 mapping from workspace-plane coordinates to image
        # coordinates (xyz->uv). The per-transformation homographies and dense remap
        # tensors are derived later, after branch and image dimensions are known.
        self.front_M = (
            None
            if front_M is None
            else np.asarray(front_M, dtype=np.float64)
        )
        self.world_fixed_img_keys = tuple(world_fixed_img_keys)
        self.front_inverse_homographies = None
        self.front_maps = {}
            
        # Set the idx value (changes depending on environment/wrapper) of the x and y observations and next_observations
        self.x_obs_idx = x_obs_idx
        self.y_obs_idx = y_obs_idx

        # Set initial fractal config values
        self.timestep = 0
        self.current_depth = 0

        self.split_method = split_method    # time / constant / never
        self.branch_method = branch_method  # constant / fractal (expansion) / contraction / disocciated

        self._handle_methods_(kwargs)

        # Instatiate homography values after _handle_methods to get keys and image sizes.
        has_front_M = self.front_M is not None
        has_world_fixed_img_keys = bool(self.world_fixed_img_keys)
        self._homography_enabled = self._homography_checks(
            has_front_M,
            has_world_fixed_img_keys,
        )

        # Warn about unused kwargs
        for k in kwargs.keys():
            print(f"\033[33mWARNING \033[0m argument \"{k}\" not used")
        
        # Prepare memory efficient storage for images in fractal: avoids storing identical image stacks once for every fractal transformation. State coordinates change between transformed copies, but image pixels remain shared.
        # copy observation space, get length, allocate circular buffer, replace orig imgs froms obs/next_obs with a frame index
        self._num_stack = None
        observation_space = copy.deepcopy(observation_space)
        next_observation_space = None

        if self.img_keys:
            self.img_buffer = {}
            next_observation_space_dict = copy.deepcopy(observation_space.spaces)

            for k in img_keys:
                img_obs_space = observation_space.spaces[k]

                if self._num_stack is None:
                    self._num_stack = img_obs_space.shape[0] # Get total number of images in experiment: wrist_1/2, front...

                # Create buffer capacity for all branches. +1 ensures partially filled final group receives storage.
                self.img_buffer_size = ((self.expected_branches + capacity - 1) // self.expected_branches) * (self._num_stack + 1) # Or, (expected_branches + capacity - 1) // expected_branches
                
                # Get buffer shape: (3, 128, 128, 3) -> [128,128,3]
                buffer_shape = list(img_obs_space.shape[1:])

                # Insert circular-buffer capacity at beginning: [128,128,3] -> [img_buffer_size, 128,128,3]
                buffer_shape.insert(0, self.img_buffer_size)


                # Instantiate img_buffer 
                self.img_buffer[k] = np.empty(buffer_shape, img_obs_space.dtype)
                
                # Define metadata for obs and next_obs spaces
                observation_space.spaces[k] = gym.spaces.Box(
                    low=float('-inf'), # TODO: change to 0
                    high=float('inf'), # TODO: change to self.img_buffer_size -1
                    shape=(), 
                    dtype=np.int32
                )

                # Remove current key. Keeping the key would cause the parent replay buffer to allocate separate storage for next images, defeating the memory optimization.
                next_observation_space_dict.pop(k) # del next_observation_space_dict[k]

            # Convert Python dict into gym.spaces.Dict()
            next_observation_space = gym.spaces.Dict(next_observation_space_dict)
            

        # Init replay buffer class
        super().__init__(
            observation_space=observation_space,
            next_observation_space=next_observation_space,
            action_space=action_space,
            capacity=capacity * self.expected_branches,
        )

        # Store each transformation index as part of the parent replay dataset
        # so its existing batched circular-insertion logic keeps this metadata
        # aligned with the corresponding transformed transition.
        if self._homography_enabled:
            self.dataset_dict["transformation_index"] = np.empty(
                self._capacity,
                dtype=np.int32,
            )

        # Precompute any symmetry transformations for branches here
        # TODO: may need to pass in argument types to determine to also include reflections/rotations/non-isometric/temporal/local/other
        self.generate_transform_deltas()
        if self._homography_enabled:
            self.generate_front_inverse_homographies()
            self.generate_front_maps()

    @property
    def num_original_transitions_inserted(self) -> int:
        """Count real environment transitions inserted before augmentation.

        ``len(self)`` reports stored replay entries. In this fractal replay
        buffer, one real transition may be expanded into many translated replay
        entries. This property keeps the unaugmented transition count available
        to learner startup logic so ``training_starts`` can mean physical actor
        experience rather than augmented copies.
        """

        return self._num_original_transitions_inserted

    def _homography_checks(
        self,
        has_front_M: bool,
        has_world_fixed_img_keys: bool,
    ) -> bool:
        if has_front_M != has_world_fixed_img_keys:
            raise ValueError(
                "front_M and world_fixed_img_keys must either both be "
                "configured or both be absent."
            )

        enabled = has_front_M and has_world_fixed_img_keys
        if not enabled:
            return False

        if self.branch_method != "constant":
            raise ValueError(
                "Homography remapping requires branch_method='constant'."
            )
        if self.front_M.shape != (3, 3):
            raise ValueError(
                f"front_M must have shape (3, 3), got {self.front_M.shape}."
            )
        if not np.all(np.isfinite(self.front_M)):
            raise ValueError("front_M must contain only finite values.")
        if np.linalg.matrix_rank(self.front_M) < 3:
            raise ValueError("front_M must be invertible.")

        missing_img_keys = set(self.world_fixed_img_keys) - set(self.img_keys)
        if missing_img_keys:
            raise ValueError(
                "world_fixed_img_keys must be included in img_keys; "
                f"missing keys: {sorted(missing_img_keys)}."
            )

        return True

    def _handle_method_arg_(self, value, method_type, method, kwargs):
        if hasattr(self, value):
            return
        assert value in kwargs.keys(), f"\033[31mERROR: \033[0m{value} must be defined for {method_type} \"{method}\""
        setattr(self, value, kwargs[value])
        del kwargs[value]

    def _handle_methods_(self, kwargs):
        
        # Initialize branch_method
        match self.branch_method:
            case "fractal":
                self._handle_method_arg_("max_depth", "branch_method", self.branch_method, kwargs)
                self._handle_method_arg_("branching_factor", "branch_method", self.branch_method, kwargs)

                self.branch = self.fractal_branch
                if not self.split_method:
                    self.split_method = "time"
                self.expected_branches = (self.branching_factor ** self.max_depth) ** 2
            
            case "contraction":
                self._handle_method_arg_("max_depth", "branch_method", self.branch_method, kwargs)
                self._handle_method_arg_("branching_factor", "branch_method", self.branch_method, kwargs)

                self.branch = self.fractal_contraction
                if not self.split_method:
                    self.split_method = "time"
                self.expected_branches = (self.branching_factor ** self.max_depth) ** 2
            
            case "linear":
                raise NotImplementedError("linear branch method is not yet implemented")
                # self.branch = self.linear_branch
                

            case "disassociated":
                self._handle_method_arg_("min_branch_count", "branch_method", self.branch_method, kwargs)
                self._handle_method_arg_("max_branch_count", "branch_method", self.branch_method, kwargs)

                if self.min_branch_count > self.max_branch_count:
                    raise ValueError(f"min_branch_count ({self.min_branch_count}) is larger than max_branch_count ({self.max_branch_count})")

                match kwargs["disassociated_type"]:
                    case "hourglass":
                        self.starting_branch_count = self.max_branch_count
                    case "octahedron":
                        self.starting_branch_count = self.min_branch_count
                    case _:
                        raise ValueError(f"incorrect value passed to disassociated_type")
                
                self.disassociated_type = kwargs["disassociated_type"]
                del kwargs["disassociated_type"]
                self.branch = self.disassociated_branch
                if not self.split_method:
                    self.split_method = "time"
                self.expected_branches = self.max_branch_count ** 2
            
            case "constant":
                self._handle_method_arg_("starting_branch_count", "branch_method", self.branch_method, kwargs)

                self.branch = self.constant_branch
                if not self.split_method:
                    self.split_method = "never"
                self.expected_branches = self.starting_branch_count ** 2
            
            case _:
                raise ValueError("incorrect value passed to branch_method")

        match self.split_method:
            case "time":
                self._handle_method_arg_("max_traj_length", "split_method", self.split_method, kwargs)
                self._handle_method_arg_("alpha", "split_method", self.split_method, kwargs)
                
                self.update_max_traj_length = True
                self.split = self.time_split 

            case "constant":
                self.split = self.constant_split
            
            case "never":
                self.split = self.never_split
                
            case _:
                raise ValueError("incorrect value passed to split_method")
        
        if hasattr(self, "starting_branch_count"):
            self.current_branch_count = self.starting_branch_count
    
    def generate_transform_deltas(self):
        """Precompute the state translations for the current branch grid.

        ``current_branch_count`` defines the number of branches (along each spatial
        axis). A count of ``b`` therefore produces ``b ** 2`` transformations,
        one for every branch in the ``k x k`` symmetric grid. The locations of the
        transformed branches follow this equation (see papers/notion Fractal Symmetry Charac for details:)

        ``transformation_i = start_loc - workspace_width/w2 + (2*i-1)*ww/2*b``
        https://bit.ly/4eXMF6W
        
        A zero delta is used for every observation component except the
        coordinates selected by indeces that require transformation: `
            `x_obs_idx`` and ``y_obs_idx``. 

        During :meth:`insert`, ``-workspace_width / 2`` is first added to those
        coordinates, then these deltas are applied. Together, those operations
        translate copies of the transition across a grid centered on the
        original position.

        The result is stored in ``self.transform_deltas`` rather than returned.
        Its shape is ``(n ** 2, observation_size)`` for an unstacked state. When
        image observations imply a frame stack, the same state translation is
        repeated across the stack, producing
        ``(n ** 2, num_stack, observation_size)``.

        This method is called at initialization and whenever the branch count
        changes, allowing insertion to apply the precomputed array directly.
        """
        
        # Pull obs out of dataset_dict that contains obs/n_obs/actions/rewards/masks/dones
        obs_state = self.dataset_dict["observations"]
        
        if self.img_keys:
            # Extract the state dictionary from dataset_dict
            obs_state = self.dataset_dict["observations"]["state"]

        # Retrieve total number of state features
        obs_size = obs_state.shape[-1]

        # For square grid, compute total num of branches
        total_branches = self.current_branch_count ** 2

        # Instantiate tensor to hold delta for each observation type. I.e. pos/vel/force/
        self.transform_deltas = np.zeros(shape=(total_branches, obs_size), dtype=np.float32)

        # Extract an index for the total num of entries in tensor
        idx = np.arange(total_branches)

        # Extract (row,col) values for x_deltas and y_deltas respectively
        x_deltas, y_deltas = np.divmod(idx, self.current_branch_count)

        # Compute transformation amount with (2*i-1)*ww/2*b as a vectorized operation
        x_deltas = (2 * x_deltas + 1) * self.workspace_width / (2 * self.current_branch_count)
        y_deltas = (2 * y_deltas + 1) * self.workspace_width / (2 * self.current_branch_count)

        # Stack on axis 1 so each row is one transformation: [delta_x, delta_y].
        base_diff = -self.workspace_width / 2.0
        self.translation_deltas_xy = np.stack(
            (
                base_diff + x_deltas,
                base_diff + y_deltas,
            ),
            axis=1,
        ).astype(np.float64)

        # Repeat delta transformation values for as many observation indices we need to do this at
        x_deltas = np.repeat(x_deltas, self.x_obs_idx.size)
        y_deltas = np.repeat(y_deltas, self.y_obs_idx.size)

        # Arrange delta transformation values in matrix form
        x_deltas = np.reshape(x_deltas, (total_branches, self.x_obs_idx.size))
        y_deltas = np.reshape(y_deltas, (total_branches, self.y_obs_idx.size))

        # Store x/y_deltas across all rows of col self.x/y_obs_idx. Will be used later in insert().
        self.transform_deltas[..., self.x_obs_idx] = x_deltas
        self.transform_deltas[..., self.y_obs_idx] = y_deltas

        # Expand transformations across stacks
        if self._num_stack:
            self.transform_deltas = np.expand_dims(self.transform_deltas, axis=1)
            self.transform_deltas = np.repeat(self.transform_deltas, self._num_stack, axis=1)

    def generate_front_inverse_homographies(self):
        """Build the destination-to-source (backwards or inverse) image homography ``H_b'' for every transform.
        (these are different from the self.front_maps created in generate_front_maps)

        ``front_M`` (front refers to the 'front' camera) is the calibration matrix that 
        maps homogeneous xyz coordinates to homogeneous image (u,v) coordinates. 
        For transform ``b``, the replay  buffer translates state by ``(delta_x_b, delta_y_b)``. 
        
        The corresponding FORWARD IMG HOMOGRAPHY is:

        ``H_b = front_M @ T_b @ inv(front_M)``,

        where ``T_b`` is the homogeneous workspace transformation. Image remapping cv2.remap()
        uses BACKWARD SAMPLING: each destination or output pixel must identify the source
        pixel from which its value is read. 
        
        The reason is that when we compute output pixels in the forward direction, sometimes,
        after computing the transformation from all input pixels, some holes will remain in the
        output image. To avoid these holes, image-warping requires us to work backwards.

        Note that when working backwards it is possible to end up with black borders. These come when
        for a number of output pixels, the equivalent image pixels was outside the original image.
        
        This method is constructed as follows:

        ``inv(H_b) = inv(front_M @ T_b @ inv(front_M)) = front_M @ inv(T_b) @ inv(front_M)``.
        See https://bit.ly/4azcCIJ

        All ``K ** 2`` inverse translations and homographies are constructed as
        BATCHED matrices. The result has shape ``(K ** 2, 3, 3)`` and is stored
        in ``self.front_inverse_homographies``.
        """

        # Get total number of translations from rows of self.translation_deltas_xy
        num_transforms = self.translation_deltas_xy.shape[0]

        # Start with one writable identity matrix per transformation. The copy
        # is required because np.broadcast_to returns a read-only view.
        inverse_translations = np.broadcast_to(
            np.eye(3, dtype=np.float64),
            (num_transforms, 3, 3),
        ).copy()

        # inv(T_b) reverses the state translation, mapping a destination
        # workspace location back to its source workspace location.
        # For the 3x3 inverse_translation matrix, subtract the \delta for all 
        # transformations (:); x or y (0 or 1), and the 3rd column (2):
        inverse_translations[:, 0, 2] = -self.translation_deltas_xy[:, 0] # dx
        inverse_translations[:, 1, 2] = -self.translation_deltas_xy[:, 1] # dy

        # Now compute inv(H)
        # Conjugating inv(T_b) by front_M expresses the inverse workspace
        # translations in front-camera pixel coordinates for backward sampling.
        front_M_inverse = np.linalg.inv(self.front_M)
        self.front_inverse_homographies = (
            self.front_M[None, :, :]
            @ inverse_translations          # @ is matrix multiplication
            @ front_M_inverse[None, :, :]
        )

    def generate_front_maps(self):
        """Build dense destination-to-source pixel maps for ``cv2.remap``.

        Let ``N = K ** 2`` be the number of transformations and
        ``P = image_height * image_width`` be the number of pixels. For each
        world-fixed image key, the destination image grid is flattened into the
        homogeneous coordinate matrix

        ``D = [[x_0, ..., x_P], [y_0, ..., y_P], [1, ..., 1]]``,

        which has shape ``(3, P)``. Batched matrix multiplication with the
        inverse image homographies produces

        ``S = self.front_inverse_homographies @ D``,

        with shape ``(N, 3, P)``. Each homogeneous source coordinate
        ``[x_h, y_h, w]`` is normalized to the image coordinate
        ``[x_h / w, y_h / w]``. These coordinates are then rearranged from
        ``(N, 2, P)`` to ``(N, image_height, image_width, 2)``.

        OpenCV uses backward sampling: the two values stored at
        ``front_maps[key][b, destination_y, destination_x]`` are the source
        ``(pixel_x, pixel_y)`` coordinates from which transformation ``b``
        should read. The final dimension of size 2 therefore represents a
        coordinate pair, not image channels.

        Each value in ``self.front_maps`` therefore has shape
        ``(K ** 2, image_height, image_width, 2)``. A loop remains only across
        camera keys because different cameras can have different resolutions.
        """
        num_transforms = self.front_inverse_homographies.shape[0]

        for key in self.world_fixed_img_keys:
            # img_buffer has shape (buffer_size, height, width, channels).
            # Selecting dimensions [1:3] extracts only the spatial resolution.
            image_height, image_width = self.img_buffer[key].shape[1:3]

            # arange constructs the one-dimensional x and y pixel axes.
            # meshgrid expands those axes into two (height, width) grids:
            # pixel_x repeats column indices and pixel_y repeats row indices.
            pixel_x, pixel_y = np.meshgrid(
                np.arange(image_width, dtype=np.float64),
                np.arange(image_height, dtype=np.float64),
            )

            # ravel flattens each grid in row-major image order. Stacking on
            # axis 0 makes one coordinate row for x, one for y, and one for the
            # homogeneous constant, producing shape (3, height * width). Each
            # column is therefore one destination pixel [x, y, 1] in homogeneous form.
            destination_pixels = np.stack(
                (
                    pixel_x.ravel(),        
                    pixel_y.ravel(),
                    np.ones(image_height * image_width, dtype=np.float64),
                ),
                axis=0,
            )

            # NumPy broadcasts (N, 3, 3) @ (3, P) into (N, 3, P), applying all
            # inverse homographies to all destination pixels without a loop
            # over transformations or pixels.
            source_pixels = (
                self.front_inverse_homographies @ destination_pixels
            )

            # source_pixels[:, :2, :] selects x_h and y_h for every transform
            # and pixel. source_pixels[:, 2:3, :] selects w while preserving a
            # singleton coordinate axis, allowing NumPy to broadcast division
            # over both x_h and y_h: [x_h, y_h] -> [x_h / w, y_h / w].
            # See https://bit.ly/4azcCIJ
            source_pixels[:, :2, :] /= source_pixels[:, 2:3, :]

            # Start with normalized coordinates shaped (N, 2, P), where P is pixels. 
            # Transpose changes this to (N, P, 2), placing [source_x, source_y] together
            # for each flattened pixel. 
            # 
            # reshape restores P to (height, width)
            # in the same row-major order used by ravel. OpenCV consumes the
            # final float32 coordinate-pair map shaped (N, height, width, 2).
            # 
            # front_maps[b, v_d, u_d, 0]  # source x-coordinate, u_s
            # front_maps[b, v_d, u_d, 1]  # source y-coordinate, v_s
            # From: front_maps[b,v_d,u_d]=u_s & v_s
            #
            # I.e. consider front_maps[0, 1, 2] = [0.5, 1.3], this says
            # to produce output pixel (x=2, y=1), sample the original image at (x=0.5, y=1.3).
            self.front_maps[key] = (
                source_pixels[:, :2, :]
                .transpose(0, 2, 1)
                .reshape(
                    num_transforms,
                    image_height,
                    image_width,
                    2,
                )
                .astype(np.float32)
            )

    def fractal_branch(self):
        '''
        Computes the number of branches for the current depth using an exponential growth rule.

        This method implements a "fractal branching" strategy, where the number of branches
        increases exponentially with depth. At each depth `d`, the number of branches is calculated as:

            num_branches = branching_factor ** current_depth

        where:
            - branching_factor: The base number of branches at each split.
            - current_depth: The current depth in the fractal tree (self.current_depth).

        Returns:
            int: The computed number of branches for the current depth.
        '''        
        # return a new number of branches = branching_factor ^ depth
        return self.branching_factor ** self.current_depth
    
    def fractal_contraction(self):
        '''
        Computes the number of branches for the current depth using a contraction rule.

        This method implements a "fractal contraction" branching strategy, where the number
        of branches decreases exponentially with depth. At each depth `d`, the number of branches
        is calculated as:

            num_branches = start_num / (branching_factor ** (d - 1))

        where:
            - start_num: The initial number of branches at depth 1.
            - branching_factor: The factor by which the number of branches contracts at each depth.
            - d: The current depth (self.current_depth).

        Returns:
            int: The computed number of branches for the current depth.
        '''

        return self.branching_factor ** (self.max_depth - self.current_depth + 1)
    
    def constant_branch(self):
        '''
        Used to create pure translations with no further branching.
        self.current_branch_count used to set the total number of transformations.
        '''
        # return current number of branches
        return self.current_branch_count
    
    def disassociated_branch(self):
        '''
        Used to create branches for disassociated fractal methods.
        self.min_branch_count specifies the mininum branch count desired during the fractal rollout
        self.max_branch_count specifies the maximum branch count desired during the fractal rollout
        self.disassociated_type specifies whether to expand and then contract or to contract and then expand
        self.steps_per_depth specifies the number of timesteps to take before splitting 
                (calculated indirectly via self.max_traj_length / self.num_depth_sectors)
        self.num_depth_sectors specifies the number of sectors the rollout should be divided into for even splitting
        '''
        if self.disassociated_type == "hourglass":
            return int((self.max_branch_count - self.min_branch_count)/(self.max_depth/2) * np.abs(self.current_depth - (self.max_depth/2)) + self.min_branch_count)
        elif self.disassociated_type == "octahedron":
            return int((self.min_branch_count - self.max_branch_count)/(self.max_depth/2) * np.abs(self.current_depth - (self.max_depth/2)) + self.max_branch_count)
        
    def linear_branch(self):
        # return a new number of branches = branches_count + n
        return self.current_branch_count + self.branching_factor
            
    def time_split(self, data_dict: DatasetDict):
        if self.timestep % (self.max_traj_length//self.max_depth) or self.current_depth >= self.max_depth:
            return False
        self.current_depth += 1
        return True 

    def constant_split(self, data_dict: DatasetDict):
        self.current_depth += 1
        return True
    
    def never_split(self, data_dict: DatasetDict):
        return False

    def insert_images(self, observation: dict, frame_index: int):
        """Inserts the original unwarped image from the replay buffer into the img_buffer.
        Note, we insert one selected temporal frame from every configured camera.

        Image observations have shape ``(T, H, W, C)``. ``frame_index``
        explicitly selects one frame along the temporal axis; the remaining
        dimensions contain the complete image. 
        
        At episode start, callers insert
        each of the ``T`` observation frames in order. At every transition,
        callers then append frame ``-1`` from the next-observation stack because
        it is the newest frame not already represented in the image ring.

        Later in insert(), the image index (self._img_insert_index) will be stored 
        in the replay buffer in lieu of the original image to save memory.

        All later transformation will maintain the img index.
        Then at the end, when the homographies are done the real images will be retrieved. 
        """
        for k in self.img_keys:
            self.img_buffer[k][self._img_insert_index_] = observation[k][
                frame_index, ...
            ]
        self._img_insert_index_ = (self._img_insert_index_ + 1) % self.img_buffer_size

    def insert(self, data: DatasetDict):
        """Insert symmetry transformations and store the augmented environment transitions.

        Copies the transition across the current symmetry grid by translating its
        selected x/y state coordinates. Other transition values are duplicated
        unchanged, while images are stored once and referenced by frame index.

        Args:
            data: Transition containing observations, next observations,
                actions, rewards, masks, and done flags.
        """

        # Create a local data_dict copy of transitions so that original "data" is not changed when transformations area done (i.e. += base_diff)
        data_dict = copy.deepcopy(data)

        # Extract obs/n_obs/actions/rewards/masks/dones
        # When images are present, state data is in key "state"
        if self.img_keys:
            obs = data_dict["observations"]["state"]
            n_obs = data_dict["next_observations"]["state"]
        else:
            obs = data_dict["observations"]
            n_obs = data_dict["next_observations"]

        actions = data_dict["actions"]
        rewards = data_dict["rewards"]
        masks = data_dict["masks"]
        dones = data_dict["dones"]

        # Update number of branches if needed (i.e. contractions/expansions/disocciated)
        if self.split(data_dict):
            temp = self.current_branch_count
            self.current_branch_count = self.branch()
            # Update transform_deltas if needed
            if temp != self.current_branch_count:
                self.generate_transform_deltas()

        # Adjust transformed x/y positions in a couple of steps. 
        # First, initialize to extreme x and y by adding -workspace_width/2.
        base_diff = -self.workspace_width/2

        obs[..., self.x_obs_idx] += base_diff
        obs[..., self.y_obs_idx] += base_diff

        n_obs[..., self.x_obs_idx] += base_diff
        n_obs[..., self.y_obs_idx] += base_diff

        # Number of transformations equal number of branches in symmetric grid
        num_transforms = self.current_branch_count ** 2

        # Create num_transforms copy based on shape for obs/actions/rewards/dones/masks
        obs_shape = np.ones(len(obs.shape) + 1, dtype=int) # i.e. np.ones(len([10,])+1)->np.ones(2,)
        obs_shape[0] = num_transforms # obs_shape -> [num_transforms 1]
        
        # Tile copies by [num_transforms,1] format
        obs = np.tile(obs, obs_shape) 
        n_obs = np.tile(n_obs, obs_shape)
        actions = np.tile(actions, (num_transforms, 1))
        rewards = np.tile(rewards, num_transforms)
        masks = np.tile(masks, num_transforms)
        dones = np.tile(dones, num_transforms)

        # In addition to base_diff done before, add actual delta transformations 
        # to correct indeces for robot (and possibly objects)
        obs += self.transform_deltas    # N transforms x obs_features
        n_obs += self.transform_deltas

        # Insert/store images in self.img_buffer via pointer mechanism
        if self.img_keys:
            if self.timestep == 0:
                # Seed the image ring with the complete T-frame observation
                # history in chronological order.
                for frame_index in range(self._num_stack): # _num_stack typically 1 for us, but could be more frames
                    self.insert_images(
                        data_dict["observations"],
                        frame_index,
                    )

            # Observation and next-observation stacks overlap by T-1 frames, so
            # only the final next-observation frame is new.
            self.insert_images(
                data_dict["next_observations"],
                -1,
            )

        # Store image index in the replay buffer
        for k in self.img_keys:
            # Store current image index in replay buffer copy, replacing original image
            data_dict["observations"][k] = (self._img_insert_index_ - 1) % len(self.img_buffer[k])

            # Create num_transform copies in a row of index and replace (lightweight)
            data_dict["observations"][k] = np.tile(data_dict["observations"][k], num_transforms)
            
            # Remove image[key] from next_obs, already in obs. Already in img_buffer via insert_images
            data_dict["next_observations"].pop(k) 

        # Pack back obs/n_obs transformations into the replay buffer
        if self.img_keys:
            data_dict["observations"]["state"] = obs
            data_dict["next_observations"]["state"] = n_obs
        else:
            data_dict["observations"] = obs
            data_dict["next_observations"] = n_obs

        # Translational transformations do not affect these
        data_dict["actions"] = actions
        data_dict["rewards"] = rewards
        data_dict["masks"] = masks
        data_dict["dones"] = dones

        # Transform copies are ordered exactly like transform_deltas.
        # As such, transformation_index identifies the correct homography
        # & dense remap grid for a given transformation.
        if self._homography_enabled:
            data_dict["transformation_index"] = np.arange(
                num_transforms,
                dtype=np.int32,
            )

        # Insert all N transformed copies across every replay field (o,no,a,r,d,m) as one
        # batch; the parent handles circular wraparound and advances its index
        # by num_transforms while preserving alignment between all fields.
        super().insert(data_dict, batch_size=num_transforms)
        self._num_original_transitions_inserted += 1

        # Reset current_depth, timestep, and max_traj_length
        self.timestep += 1
        if data_dict["dones"][0]:
            self.current_depth = 0

            # Adaptive max_traj_length estimate - useful in time-based split methods.
            if self.update_max_traj_length:
                # Update the trajectory-length estimate with an exponential
                # moving average: alpha weights this completed episode's
                # observed length, while 1-alpha retains the prior estimate.
                self.max_traj_length = int(self.timestep * self.alpha + self.max_traj_length * (1 - self.alpha))
            self.timestep = 0
    
    def sample(
        self, 
        batch_size: int, 
        keys: Optional[Iterable[str]] = None, 
        indx: Optional[np.ndarray] = None, 
        pack_obs_and_next_obs: bool = False,
    ) -> frozen_dict.FrozenDict:                # immutable
        """Samples from the replay buffer.

        When ``indx`` is not provided, ``self.np_random`` samples stored
        transitions uniformly with replacement. Call :meth:`seed` for
        reproducible batches. Sampling is uniform over transformed branch
        copies, so timesteps that generated more copies have proportionally
        greater representation.

        With images, the parent replay dataset does not contain complete image
        observations. Its ``observations`` field contains state arrays and
        INTEGER POINTERS into ``self.img_buffer``. Consequently, this method
        temporarily removes the top-level ``"observations"`` key only from the
        list passed to the parent sampler. It does not remove observation data
        from ``self.dataset_dict``.

        After the parent samples the other public transition fields, this method
        directly samples observation state from
        ``self.dataset_dict["observations"]`` using the same replay indices. It
        then uses the aligned image pointers to reconstruct frame stacks from
        ``self.img_buffer``. State and reconstructed images are placed back
        under ``batch["observations"]``, so ``"observations"`` remains part of
        the returned learner batch. The internal ``"transformation_index"`` is
        used only to select an image remap and is not returned.

        A reconstructed image window has shape
        ``(batch_size, stack_size + 1, height, width, channels)``. For a
        world-fixed camera, the batch and temporal dimensions are temporarily
        flattened so OpenCV can remap one image per call. Each sampled
        transition's transformation index is repeated ``stack_size + 1`` times,
        ensuring every frame in its combined observation/next-observation
        window uses the same spatial transformation. The remapped images are
        restored to their original five-dimensional shape before the overlapping
        observation and next-observation stacks are separated. Robot-mounted
        cameras that are absent from ``world_fixed_img_keys`` bypass remapping.

        Args:
            batch_size: Minibatch size.
            keys: Keys to sample.
            indx: Take indices instead of sampling.
            pack_obs_and_next_obs: whether to pack img and next_img into one image.
                It's useful when they have overlapping frames.

        Returns:
            A frozen dictionary.
        """
        # If no images, sample normally
        if not self.img_keys:
            return super().sample(batch_size, keys, indx)
        
        # Generate random indexes for sampling
        if indx is None:
            if hasattr(self.np_random, "integers"):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

            for i in range(batch_size):
                while indx[i] >= self._size:
                    if hasattr(self.np_random, "integers"):
                        indx[i] = self.np_random.integers(len(self))
                    else:
                        indx[i] = self.np_random.randint(len(self))
        else:
            raise NotImplementedError()

        # Use the sampled replay-entry indices to retrieve the transformation
        # metadata aligned with each transition. This remains local to sample()
        # and will select the corresponding dense image map.
        transformation_indices = None
        if self._homography_enabled:
            transformation_indices = self.dataset_dict[
                "transformation_index"
            ][indx]

        # Sample w/o images
        # Top-level keys describe the fields of a complete transition:
        # observations, next_observations, actions, rewards, masks, and dones.
        # Camera names such as "front" are nested below the "observations" key.
        # Include every public transition field by default, but exclude the
        # internal transformation_index because it is replay-buffer metadata,
        # not an input expected by the learner.
        if keys is None:
            keys = [
                key
                for key in self.dataset_dict.keys()
                if key != "transformation_index"
            ]
        else:
            # Collect all top-level keys except "transformation_index"
            keys = [
                key
                for key in keys
                if key != "transformation_index"
            ]
            assert "observations" in keys

        # Remove observations only from this local list of keys passed to the
        # parent sampler; self.dataset_dict is unchanged. The parent would
        # otherwise return integer image pointers as though they were images.
        # Observation state and image stacks are reconstructed manually below
        # and restored under the public batch["observations"] field.
        keys.remove("observations")

        # Returns a frozen dictionary that is immutable
        batch = super().sample(batch_size, keys, indx)

        # Convert the frozen dict (from Flax lib) into a mutable Python dict to add reconstructed images.
        batch = batch.unfreeze()

        # Extract ["state"] observation keys, but not image keys (i.e. 'wrist' and 'front')
        obs_keys = self.dataset_dict["observations"].keys()
        obs_keys = list(obs_keys)
        for k in self.img_keys:
            obs_keys.remove(k)

        # Create new dict for manually reconstructed observations.
        batch["observations"] = {}
        
        # Sample non-image observation data, aka state with the same indices used for actions
        for k in obs_keys:            
            batch["observations"][k] = _sample( 
                self.dataset_dict["observations"][k], indx
            )

        # Reconstruct image observations manually. 
        # Each stored observation value is a pointer into the image buffer.
        for k in self.img_keys:

            # Retrieve specific camera frame
            # Note: obs_imgs holds a lightweight indexing view over the original frame buffer
            obs_imgs = self.img_buffer[k]

            # Create overlapping windows/arrays over the image buffer without COPYING the underlying frames.
            # img_buffer = [f0, f1, f2, f3, f4, f5] -- 4 --> 
            # obs_img[0]=[f0, f1, f2, f3]; obs_img[1]=[f1, f2, f3, f4]; obs_img[2]=[f2, f3, f4, f5]
            obs_imgs = np.lib.stride_tricks.sliding_window_view(
                obs_imgs, 
                self._num_stack + 1, 
                axis=0
            )

            # Extract specific images according to batch indeces:
            # Calculate each window’s starting frame
            obs_imgs = obs_imgs[self.dataset_dict["observations"][k][indx] - self._num_stack]
            
            # transpose from (Batch, Height, Width, Channels(RGB), Time/Sequence) to (B, T, H, W, C) to follow JAXRL_M convention
            obs_imgs = obs_imgs.transpose((0, 4, 1, 2, 3))

            # Only transform global images, not wrist ones.
            if k in self.world_fixed_img_keys:
                # obs_imgs has shape (B, T+1, H, W, C). T+1 is the combined
                # window containing both overlapping T-frame observation and
                # next-observation stacks.
                batch_size_actual, stack_plus_one = obs_imgs.shape[:2]

                # Collapse batch and time into one image dimension:
                # (B, T+1, H, W, C) -> (B*(T+1), H, W, C).
                # CPU cv2.remap accepts one image per call.
                flat_images = obs_imgs.reshape(
                    batch_size_actual * stack_plus_one,
                    *obs_imgs.shape[2:],
                )

                # Each replay entry has one transformation index. Repeat it for
                # every frame in that entry's T+1 window so the index order
                # matches the flattened image order exactly.
                flat_transform_indices = np.repeat(
                    transformation_indices,
                    stack_plus_one,
                )

                # Allocate output
                # Use the same shape and dtype as the flattened
                # source images. Every entry is filled by cv2.remap below.
                flat_output = np.empty_like(flat_images)

                # OpenCV's CPU remap API operates on one image at a time. 
                # The expensive pixel-coordinate calculation is not repeated here:
                # front_maps already stores a dense (H, W, 2) source-coordinate
                # map for every transformation.
                for image_idx, transform_idx in enumerate(flat_transform_indices):

                    flat_output[image_idx] = cv2.remap(
                        flat_images[image_idx],
                        self.front_maps[k][transform_idx], # map1 combines src(x,y) coords, no need for map2 arg                        
                        None,                                            
                        interpolation=cv2.INTER_LINEAR,     # Homography coordinates are generally fractional; bilinear interpolation estimates their pixel values.                                                
                        borderMode=cv2.BORDER_REPLICATE,    # Coords outside the source image use the nearest edge pixel instead of introducing an arbitrary color.
                    )

                # Restore (B, T+1, H, W, C). This does not create an additional
                # K**2 dimension: each of the B sampled replay entries selected
                # exactly one transformation map.
                obs_imgs = flat_output.reshape(obs_imgs.shape)

            if pack_obs_and_next_obs:
                batch["observations"][k] = obs_imgs
            else:
                # Place in batch all but the last observations image from the image stack :-1 for 
                batch["observations"][k] = obs_imgs[:, :-1, ...]

                if "next_observations" in keys:
                    # Place in batch all from 2nd index till end of image stack 1:
                    batch["next_observations"][k] = obs_imgs[:, 1:, ...]

        return frozen_dict.freeze(batch)
