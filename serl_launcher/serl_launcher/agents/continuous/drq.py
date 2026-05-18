import copy
from collections import OrderedDict
from functools import partial
from typing import Dict, Iterable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import frozen_dict

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack, concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop


class DrQAgent(SACAgent):
    """SAC-style agent specialized for pixel observations with DrQ augmentation.

    DrQ (Data-regularized Q-learning) improves visual RL by applying image
    augmentations during training so the critic learns features that are less
    sensitive to small visual changes (for example camera jitter or lighting).
    """

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Models
        actor_def: nn.Module,
        critic_def: nn.Module,
        temperature_def: nn.Module,
        # Optimizer
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        temperature_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        # Algorithm config
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        entropy_per_dim: bool = False,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = ("image",),
    ):
        """Construct a DrQAgent from already-built actor/critic/temperature modules.

        This method is the low-level constructor used after network definitions
        are prepared. It initializes parameters, optimizers, and target network
        state, then stores algorithm hyperparameters in ``config``.

        Args:
            rng: JAX random key used for parameter/state initialization.
            observations: Example observation batch used to initialize module
                shapes.
            actions: Example action batch used to initialize critic/action shapes.
            actor_def: Actor network module.
            critic_def: Critic network module.
            temperature_def: Learnable temperature module for entropy weighting.
            actor_optimizer_kwargs: Optimizer settings for actor updates.
            critic_optimizer_kwargs: Optimizer settings for critic updates.
            temperature_optimizer_kwargs: Optimizer settings for temperature.
            discount: TD discount factor.
            soft_target_update_rate: Polyak averaging factor for target critic.
            target_entropy: Target policy entropy. If ``None``, uses a default
                based on action dimension.
            entropy_per_dim: Unsupported in this class (kept for API parity).
            backup_entropy: Whether to include entropy term in critic target.
            critic_ensemble_size: Number of critic heads in the ensemble.
            critic_subsample_size: Optional REDQ-style subset size for target min.
            image_keys: Observation keys treated as pixel tensors.

        Returns:
            Initialized ``DrQAgent`` with online and target parameters.
        """
        networks = {
            "actor": actor_def,
            "critic": critic_def,
            "temperature": temperature_def,
        }

        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=[observations],
            critic=[observations, actions],
            temperature=[],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        # Config
        assert not entropy_per_dim, "Not implemented"
        if target_entropy is None:
            target_entropy = -actions.shape[-1] / 2

        return cls(
            state=state,
            config=dict(
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                target_entropy=target_entropy,
                backup_entropy=backup_entropy,
                image_keys=image_keys,
            ),
        )

    @classmethod
    def create_drq(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_type: str = "small",
        shared_encoder: bool = True,
        use_proprio: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("image",),
        **kwargs,
    ):
        """Build and initialize a pixel-based DrQ agent from high-level choices.

        This helper chooses an image encoder family, wraps image/proprio inputs,
        constructs actor and critic networks, and then calls ``create`` to
        initialize training state.

        Designed for students:
        - Think of this as an "agent factory" that wires all pieces together.
        - ``encoder_type`` picks how images become feature vectors.
        - The critic is an ensemble (multiple Q-networks) for stability.
        - DrQ augmentation is configured later in ``data_augmentation_fn`` and
          used during updates.

        Args:
            rng: Random key used for model initialization.
            observations: Example observation used for shape inference.
            actions: Example action batch used for shape inference.
            encoder_type: One of ``"small"``, ``"resnet"``, or
                ``"resnet-pretrained"``.
            shared_encoder: Kept for API compatibility; current implementation
                uses a shared encoder instance for actor and critic.
            use_proprio: If ``True``, include non-image state features.
            critic_network_kwargs: MLP kwargs for critic backbone.
            policy_network_kwargs: MLP kwargs for policy backbone.
            policy_kwargs: Extra kwargs forwarded to ``Policy``.
            critic_ensemble_size: Number of critic ensemble heads.
            critic_subsample_size: Optional number of heads sampled for targets.
            temperature_init: Initial value for entropy temperature.
            image_keys: Keys in observation dict that contain image stacks.
            **kwargs: Forwarded to ``create`` (e.g., discount, optimizers).

        Returns:
            A fully initialized ``DrQAgent`` ready for training.
        """
        # Ensure the final MLP layer is active so the heads can freely shape
        # outputs before final linear projections in Policy/Critic modules.
        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        # Choose the visual encoder architecture.
        # Each image key gets its own named encoder module.
        if encoder_type == "small":
            from serl_launcher.vision.small_encoders import SmallEncoder

            encoders = {
                image_key: SmallEncoder(
                    features=(32, 64, 128, 256),
                    kernel_sizes=(3, 3, 3, 3),
                    strides=(2, 2, 2, 2),
                    padding="VALID",
                    pool_method="avg",
                    bottleneck_dim=256,
                    spatial_block_size=8,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet":
            # Train a lightweight ResNet-10 style encoder from scratch.
            from serl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            # Use a frozen pretrained ResNet trunk, then task-specific pooling/head.
            from serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            # Shared frozen feature extractor used by every image stream.
            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )
            encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        # Wrap image encoders so the model can:
        # - merge multiple camera streams
        # - optionally concatenate proprioception
        # - handle frame-stacked inputs consistently
        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        # Actor and critic currently share the same encoder wrapper instance.
        encoders = {
            "critic": encoder_def,
            "actor": encoder_def,
        }

        # Build critic backbone and replicate it into an ensemble.
        # Ensemble critics reduce overestimation and improve stability.
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        # Critic consumes encoded observations + actions.
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")

        # Policy consumes encoded observations and outputs an action distribution.
        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
            name="actor",
        )

        # Entropy temperature alpha (>= 0) is learned with a constrained module.
        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        # Delegate common initialization (params, optimizers, target params).
        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            **kwargs,
        )

        # For pretrained mode, load frozen ResNet-10 weights after module setup.
        if encoder_type == "resnet-pretrained":
            from serl_launcher.utils.train_utils import load_resnet10_params

            agent = load_resnet10_params(agent, image_keys)

        return agent

    def data_augmentation_fn(self, rng, observations):
        """Apply DrQ image augmentation (random crop) to each configured image key.

        The augmentation is applied to pixel observations only; non-image fields
        in ``observations`` are preserved. This regularizes visual features so
        Q-learning is less sensitive to small image-level perturbations.

        Args:
            rng: Random key controlling crop offsets.
            observations: Observation dict/FrozenDict containing pixel stacks.

        Returns:
            A copy of ``observations`` where each image tensor in ``image_keys``
            is replaced by a randomly cropped version.
        """
        # Iterate over every configured camera/image input.
        for pixel_key in self.config["image_keys"]:
            # Replace this key with an augmented tensor while leaving all other
            # observation entries unchanged.
            #
            # padding=4:
            #   First pad the image by 4 pixels on each side (edge padding), then
            #   crop back to original size at a random offset.
            # num_batch_dims=2:
            #   Data is usually [batch, stack, H, W, C], so we treat the first
            #   two axes as batch-like dimensions when vectorizing crops.
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    @partial(jax.jit, static_argnames=("utd_ratio", "pmap_axis"))
    def update_high_utd(
        self,
        batch: Batch,
        *,
        utd_ratio: int,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["DrQAgent", dict]:
        """JIT-compiled high-UTD training step with DrQ augmentation.

        Steps:
        1. Ensure packed replay samples are unpacked when needed.
        2. Apply random-crop augmentation to current and next observations.
        3. Run the parent SAC high-UTD update:
           - multiple critic updates
           - one actor/temperature update

        Args:
            batch: Training batch sampled from replay.
            utd_ratio: Number of critic updates per actor update.
            pmap_axis: Optional PMAP axis name for distributed reductions.

        Returns:
            Tuple of updated agent and merged logging metrics.
        """
        new_agent = self
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        rng = new_agent.state.rng
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = self.data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = self.data_augmentation_fn(next_obs_rng, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
            }
        )

        new_state = self.state.replace(rng=rng)

        new_agent = self.replace(state=new_state)
        return SACAgent.update_high_utd(
            new_agent, batch, utd_ratio=utd_ratio, pmap_axis=pmap_axis
        )

    @partial(jax.jit, static_argnames=("pmap_axis",))
    def update_critics(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["DrQAgent", dict]:
        """Run a critic-only update with DrQ augmentation.

        This method is useful when training with a critic:actor update ratio
        greater than 1. It augments observations, updates only critic parameters
        (and target critic via parent update logic), and returns critic metrics.

        Args:
            batch: Training batch from replay (packed or unpacked).
            pmap_axis: Optional PMAP axis name for distributed reductions.

        Returns:
            Tuple of updated agent and critic-only logging info.
        """
        new_agent = self
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        rng = new_agent.state.rng
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = self.data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = self.data_augmentation_fn(next_obs_rng, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
            }
        )

        new_state = self.state.replace(rng=rng)
        new_agent = self.replace(state=new_state)
        new_agent, critic_infos = new_agent.update(
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=frozenset({"critic"}),
        )
        del critic_infos["actor"]
        del critic_infos["temperature"]

        return new_agent, critic_infos
