from functools import partial
from typing import Optional

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.common.common import default_init
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.jax_utils import next_rng


class ValueCritic(nn.Module):
    """State-value network that predicts ``V(s)`` from observations.

    In actor-critic RL, a value critic estimates how good a state is on average,
    independent of a specific action. This module maps observations to a single
    scalar per sample:
    - input: observation tensor(s)
    - output: one value estimate per observation (shape ``[batch]``)

    Architecture used here:
    1. ``encoder`` transforms raw observations into learned features.
    2. ``network`` processes those features into a hidden representation.
    3. A final linear layer (``Dense(1)``) produces the scalar value.

    Initialization behavior:
    - If ``init_final`` is provided, the final layer kernel is initialized
      uniformly in ``[-init_final, init_final]`` for tighter initial output scale.
    - Otherwise, the project default initializer is used.

    Notes for students:
    - ``train`` is passed through so submodules can switch behavior (for example,
      dropout/batch-norm if those are used in ``encoder``/``network``).
    - The returned tensor is ``squeeze``d on the last axis, converting
      ``[..., 1]`` to ``[...]`` for convenience in losses.
    """
    # Encodes raw observations into feature vectors.
    encoder: nn.Module
    # Backbone network that maps encoded features to a critic hidden representation.
    network: nn.Module
    # Optional bound for uniform init of the final value head weights.
    init_final: Optional[float] = None

    # Define submodules inline in __call__; Flax registers params automatically.
    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        outputs = self.network(self.encoder(observations), train=train)
        if self.init_final is not None:
            # Use a narrow, explicit final-layer init when requested.
            value = nn.Dense(
                1,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            # Fall back to the default project initializer.
            value = nn.Dense(1, kernel_init=default_init())(outputs)
        return jnp.squeeze(value, -1)


def multiple_action_q_function(forward):
    """Decorator that makes a critic forward pass support multiple actions/state.

    Args:
        forward: The original critic method (typically ``__call__``) that maps
            ``(self, observations, actions, **kwargs)`` to Q-values for one
            action per state.

    Returns:
        A wrapped function that:
        - calls ``forward`` directly when ``actions`` is 2D (single action/state)
        - vmaps ``forward`` over the action axis when ``actions`` is 3D
          (multiple candidate actions/state), returning stacked Q-values.
          vmap ensures that 
    """
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions, **kwargs):
        if jnp.ndim(actions) == 3:
            q_values = jax.vmap(
                lambda a: forward(self, observations, a, **kwargs),
                in_axes=1,
                out_axes=-1,
            )(actions)
        else:
            q_values = forward(self, observations, actions, **kwargs)
        return q_values

    return wrapped


class Critic(nn.Module):
    encoder: Optional[nn.Module]
    network: nn.Module
    init_final: Optional[float] = None

    # Define submodules inline in __call__; Flax registers params automatically.
    @nn.compact
    @multiple_action_q_function
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, train: bool = False
    ) -> jnp.ndarray:
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations)

        inputs = jnp.concatenate([obs_enc, actions], -1)
        outputs = self.network(inputs, train=train)
        if self.init_final is not None:
            value = nn.Dense(
                1,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            value = nn.Dense(1, kernel_init=default_init())(outputs)
        return jnp.squeeze(value, -1)


class DistributionalCritic(nn.Module):
    """Distributional variant of ``Critic`` that predicts a value distribution.

    Unlike ``Critic`` (which outputs one scalar Q-value per state-action pair),
    this module outputs:
    - ``logits`` over ``num_atoms`` support points
    - the corresponding ``atoms`` linearly spaced in ``[q_low, q_high]``

    The expected Q-value can be recovered downstream from these distributional
    outputs, while training can use distributional RL losses.
    """
    encoder: Optional[nn.Module]
    network: nn.Module
    q_low: float
    q_high: float
    num_atoms: int = 51
    init_final: Optional[float] = None

    # Define submodules inline in __call__; Flax registers params automatically.
    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, train: bool = False
    ) -> jnp.ndarray:
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations)

        inputs = jnp.concatenate([obs_enc, actions], -1)
        outputs = self.network(inputs, train=train)
        if self.init_final is not None:
            logits = nn.Dense(
                self.num_atoms,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final), # uniform weight initialization. gives a strict bound on weights vs gaussian. early atom logits/Q outputs small for stable training.
            )(outputs)
        else:
            logits = nn.Dense(self.num_atoms, kernel_init=default_init())(outputs)

        atoms = jnp.linspace(self.q_low, self.q_high, self.num_atoms)
        atoms = jnp.broadcast_to(atoms, logits.shape)

        return logits, atoms


class ContrastiveCritic(nn.Module):
    """Contrastive value model using state-action and goal embeddings.

    Unlike ``Critic`` (scalar Q per state-action) and ``DistributionalCritic``
    (distribution over fixed value atoms), this module does not predict Q
    directly. It learns embeddings for:
    - ``(state, action)`` via ``sa_net``
    - ``goal`` via ``g_net``

    It returns their pairwise similarity matrix (dot products). With
    ``twin_q=True``, a second independent similarity head is stacked on the
    last axis.
    """
    encoder: nn.Module
    sa_net: nn.Module
    g_net: nn.Module
    repr_dim: int = 16
    twin_q: bool = True
    sa_net2: Optional[nn.Module] = None
    g_net2: Optional[nn.Module] = None
    init_final: Optional[float] = None

    # Define submodules inline in __call__; Flax registers params automatically.
    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, train: bool = False
    ) -> jnp.ndarray:
        obs_goal_encoding = self.encoder(observations)
        encoding_dim = obs_goal_encoding.shape[-1] // 2
        obs_encoding, goal_encoding = (
            obs_goal_encoding[..., :encoding_dim],
            obs_goal_encoding[..., encoding_dim:],
        )

        if self.init_final is not None:
            kernel_init = partial(
                nn.initializers.uniform, -self.init_final, self.init_final
            )
        else:
            kernel_init = default_init

        sa_inputs = jnp.concatenate([obs_encoding, actions], -1)
        sa_repr = self.sa_net(sa_inputs, train=train)
        sa_repr = nn.Dense(self.repr_dim, kernel_init=kernel_init())(sa_repr)
        g_repr = self.g_net(goal_encoding, train=train)
        g_repr = nn.Dense(self.repr_dim, kernel_init=kernel_init())(g_repr)
        outer = jnp.einsum("ik,jk->ij", sa_repr, g_repr)

        if self.twin_q:
            sa_repr2 = self.sa_net2(sa_inputs, train=train)
            sa_repr2 = nn.Dense(self.repr_dim, kernel_init=kernel_init())(sa_repr2)
            g_repr2 = self.g_net2(goal_encoding, train=train)
            g_repr2 = nn.Dense(self.repr_dim, kernel_init=kernel_init())(g_repr2)
            outer2 = jnp.einsum("ik,jk->ij", sa_repr2, g_repr2)

            outer = jnp.stack([outer, outer2], axis=-1)

        return outer


def ensemblize(cls, num_qs, out_axes=0):
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
    )


class Policy(nn.Module):
    """Gaussian actor that outputs an action distribution from observations.

    This is the stochastic policy used by actor-critic methods (for example,
    SAC-style training). Given an observation, the module predicts:
    - ``means``: the center of the action distribution for each action dimension
    - ``stds``: the spread (uncertainty / exploration scale) per action dimension

    The policy returns a distribution object, not a sampled action. Downstream
    code can then sample actions, compute log-probabilities, or evaluate entropy.

    High-level flow:
    1. Optionally encode observations with ``encoder``.
    2. Pass features through ``network``.
    3. Predict action means with a linear layer.
    4. Compute standard deviations using one of ``std_parameterization`` modes.
    5. Clip stds to ``[std_min, std_max]`` and scale by ``sqrt(temperature)``.
    6. Return either a Gaussian distribution or a tanh-squashed Gaussian.

    Std parameterization modes:
    - ``"exp"``:
        Predict ``log_stds`` from the network and set ``stds = exp(log_stds)``.
        Common choice; ensures positivity while letting the network adapt stds by
        state.
    - ``"softplus"``:
        Predict unconstrained values and map with ``softplus`` for positive stds.
        Similar goal to ``exp`` but with different gradients/saturation behavior.
    - ``"uniform"``:
        Use one learned vector ``log_stds`` shared across all states (state-
        independent std), then exponentiate.
    - ``"fixed"``:
        Do not learn std from this module; use ``fixed_std`` provided externally.

    Action bounds:
    - If ``tanh_squash_distribution=False``, output is an unconstrained Gaussian
        (``distrax.MultivariateNormalDiag``).
    - If ``True``, output is transformed through tanh
        (``TanhMultivariateNormalDiag``), commonly used for bounded actions.

    Notes for students:
    - ``init_final`` is currently a config field but is not used in this class's
        final layers.
    - ``temperature`` controls exploration scale at runtime by multiplying std by
        ``sqrt(temperature)``. Larger temperature => broader sampling.
    """
    encoder: Optional[nn.Module]
    network: nn.Module
    action_dim: int
    init_final: Optional[float] = None
    std_parameterization: str = "exp"  # "exp", "softplus", "fixed", or "uniform"
    std_min: Optional[float] = 1e-5
    std_max: Optional[float] = 10.0
    tanh_squash_distribution: bool = False
    fixed_std: Optional[jnp.ndarray] = None

    # Define submodules inline in __call__; Flax registers params automatically.
    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0, train: bool = False
    ) -> distrax.Distribution:
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations, train=train, stop_gradient=True)

        outputs = self.network(obs_enc, train=train)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        if self.fixed_std is None:
            if self.std_parameterization == "exp":
                log_stds = nn.Dense(self.action_dim, kernel_init=default_init())(
                    outputs
                )
                stds = jnp.exp(log_stds)
            elif self.std_parameterization == "softplus":
                stds = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
                stds = nn.softplus(stds)
            elif self.std_parameterization == "uniform":
                log_stds = self.param(
                    "log_stds", nn.initializers.zeros, (self.action_dim,)
                )
                stds = jnp.exp(log_stds)
            else:
                raise ValueError(
                    f"Invalid std_parameterization: {self.std_parameterization}"
                )
        else:
            assert self.std_parameterization == "fixed"
            stds = jnp.array(self.fixed_std)

        # Clip stds to avoid numerical instability
        # For a normal distribution under MaxEnt, optimal std scales with sqrt(temperature)
        stds = jnp.clip(stds, self.std_min, self.std_max) * jnp.sqrt(temperature)

        if self.tanh_squash_distribution:
            distribution = TanhMultivariateNormalDiag(
                loc=means,
                scale_diag=stds,
            )
        else:
            distribution = distrax.MultivariateNormalDiag(
                loc=means,
                scale_diag=stds,
            )

        return distribution


class TanhMultivariateNormalDiag(distrax.Transformed):
    """Diagonal Gaussian policy transformed by tanh (and optional rescaling).

    This class wraps a base multivariate normal distribution and applies a
    bijective transform to produce bounded actions. It is commonly used in
    continuous-control RL because:
    - a Gaussian is easy to optimize in latent action space
    - tanh squashing keeps sampled actions bounded
    - log-probabilities remain correct through change-of-variables

    Conceptually, it does:
    1. Sample latent action ``u ~ Normal(loc, scale_diag)``.
    2. Squash with ``tanh`` to get values in ``(-1, 1)``.
    3. If ``low`` and ``high`` are provided, map from ``(-1, 1)`` into
       ``[low, high]`` elementwise.

    Because this is a ``distrax.Transformed`` distribution, ``log_prob`` and
    sampling automatically account for the bijector Jacobian (including the
    custom affine rescale Jacobian when bounds are provided).

    Args:
        loc: Mean vector of the base diagonal Gaussian.
        scale_diag: Per-dimension standard deviations of the base Gaussian.
        low: Optional lower action bounds. Must broadcast with action shape.
        high: Optional upper action bounds. Must broadcast with action shape.
            If either ``low`` or ``high`` is missing, no extra rescaling is
            applied and outputs stay in ``(-1, 1)`` after tanh.
    """
    def __init__(
        self,
        loc: jnp.ndarray,
        scale_diag: jnp.ndarray,
        low: Optional[jnp.ndarray] = None,
        high: Optional[jnp.ndarray] = None,
    ):
        distribution = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        layers = []

        if not (low is None or high is None):

            def rescale_from_tanh(x):
                x = (x + 1) / 2  # (-1, 1) => (0, 1)
                return x * (high - low) + low

            def forward_log_det_jacobian(x):
                high_ = jnp.broadcast_to(high, x.shape)
                low_ = jnp.broadcast_to(low, x.shape)
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

            layers.append(
                distrax.Lambda(
                    rescale_from_tanh,
                    forward_log_det_jacobian=forward_log_det_jacobian,
                    event_ndims_in=1,
                    event_ndims_out=1,
                )
            )

        layers.append(distrax.Block(distrax.Tanh(), 1))

        bijector = distrax.Chain(layers)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    def stddev(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.stddev())
