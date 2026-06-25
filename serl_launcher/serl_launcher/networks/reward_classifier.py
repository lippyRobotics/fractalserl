import os
import pickle as pkl
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.training import checkpoints
import optax
from typing import Callable, Dict, List, Optional


from serl_launcher.vision.resnet_v1 import resnetv1_configs, PreTrainedResNetEncoder
from serl_launcher.common.encoding import EncodingWrapper
from flax.core.frozen_dict import freeze, unfreeze


class BinaryClassifier(nn.Module):
    encoder_def: nn.Module
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x, train=False):
        x = self.encoder_def(x, train=train)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(0.1)(x, deterministic=not train)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


def create_classifier(
    key: jnp.ndarray,
    sample: Dict,
    image_keys: List[str],
    pretrained_encoder_path: str = "./resnet10_params.pkl",
    use_proprio: bool = False,
):
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
    encoder_def = EncodingWrapper(
        encoder=encoders,
        use_proprio=use_proprio,
        enable_stacking=True,
        image_keys=image_keys,
    )

    classifier_def = BinaryClassifier(encoder_def=encoder_def)
    params = classifier_def.init(key, sample)["params"]
    classifier_def = BinaryClassifier(encoder_def=encoder_def)
    params = freeze(params)
    classifier = TrainState.create(
        apply_fn=classifier_def.apply,
        params=params,
        tx=optax.adam(learning_rate=1e-4),
    )

    if not os.path.exists(pretrained_encoder_path):
        cached_path = os.path.expanduser("~/.serl/resnet10_params.pkl")
        if (
            os.path.basename(pretrained_encoder_path) == "resnet10_params.pkl"
            and os.path.exists(cached_path)
        ):
            pretrained_encoder_path = cached_path

    if not os.path.exists(pretrained_encoder_path):
        print(
            f"Warning: pretrained encoder params not found at "
            f"{pretrained_encoder_path}. Using initialized classifier weights."
        )
        return classifier

    with open(pretrained_encoder_path, "rb") as f:
        encoder_params = pkl.load(f)
    param_count = sum(x.size for x in jax.tree.leaves(encoder_params))
    print(
        f"Loaded {param_count/1e6}M parameters from ResNet-10 pretrained on ImageNet-1K"
    )

    new_params = classifier.params.unfreeze()
    for image_key in image_keys:
        if "pretrained_encoder" in new_params["encoder_def"][f"encoder_{image_key}"]:
            for k in new_params["encoder_def"][f"encoder_{image_key}"][
                "pretrained_encoder"
            ]:
                if k in encoder_params:
                    new_params["encoder_def"][f"encoder_{image_key}"][
                        "pretrained_encoder"
                    ][k] = encoder_params[k]
                    print(f"replaced {k} in encoder_{image_key}")

    new_params = freeze(new_params)
    classifier = classifier.replace(params=new_params)
    return classifier


def load_classifier_func(
    key: jnp.ndarray,
    sample: Dict,
    image_keys: List[str],
    checkpoint_path: str,
    step: Optional[int] = None,
) -> Callable[[Dict], jnp.ndarray]:
    """
    Return: a function that takes in an observation
            and returns the logits of the classifier.
    """
    classifier = create_classifier(key, sample, image_keys)
    classifier = checkpoints.restore_checkpoint(
        checkpoint_path,
        target=classifier,
        step=step,
    )
    func = lambda obs: classifier.apply_fn(
        {"params": classifier.params}, obs, train=False
    )
    func = jax.jit(func)
    return func
