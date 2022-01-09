import gym
import torch
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from typing import Any, Dict, List, Optional, Type, Union

from torch.nn import LayerNorm

from custom_policies.transformer_encoder_layer import TransformerEncoderLayer, TransformerEncoder

from custom_policies.transformer_pytorch import PositionalEncoding, CustomCombinedExtractor

RESIDUE_LETTERS = [
    'R', 'H', 'K',
    'D', 'E',
    'S', 'T', 'N', 'Q',
    'C', 'G', 'P',
    'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'empty'
]

class MultipleLayerExtractor(CustomCombinedExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, embedding_dim=16, num_heads=2, num_layers=2):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)
        embed_dim = embedding_dim
        num_heads = num_heads

        self.pe = PositionalEncoding(embed_dim, 0, max_len=32)

        self.transformerEncoderLayer = TransformerEncoderLayer(embed_dim, num_heads,
                                                               dim_feedforward=64)

        self.transformerEncoderStandardLayer = TransformerEncoderLayer(
            embed_dim,
            num_heads, 32, 0,
            torch.nn.functional.relu,
            1e-5, True
        )
        encoder_norm = LayerNorm(embed_dim, eps=1e-5)
        self.encoder = TransformerEncoder(
            self.transformerEncoderLayer,
            self.transformerEncoderStandardLayer,
            num_layers=num_layers, norm=encoder_norm
        )

        self.value_key = nn.Conv1d(observation_space['torsion_angles'].shape[1], embed_dim, (1,))
        self.query = nn.Embedding(len(RESIDUE_LETTERS) + 2, embed_dim)

        self._features_dim = embed_dim * observation_space['torsion_angles'].shape[0] + 2


class ActorCriticTransformerMultipleLayersPolicy(ActorCriticPolicy):

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = MultipleLayerExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(ActorCriticTransformerMultipleLayersPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )