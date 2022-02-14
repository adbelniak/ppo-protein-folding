import gym
import torch
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)
from torch import nn, transpose

from stable_baselines3.common.maskable.policies import MaskableActorCriticPolicy

from stable_baselines3.common.type_aliases import Schedule
from typing import Any, Dict, List, Optional, Type, Union

from torch.nn import LayerNorm, TransformerEncoderLayer as StandardTransformerEncoderLayer, TransformerEncoder



from custom_policies.transformer_pytorch import PositionalEncoding, CustomCombinedExtractor

RESIDUE_LETTERS = [
    'R', 'H', 'K',
    'D', 'E',
    'S', 'T', 'N', 'Q',
    'C', 'G', 'P',
    'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'empty'
]

class JointInputFeatureExtractor(CustomCombinedExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, embedding_dim=16, num_heads=2, num_layers=2):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)
        embed_dim = embedding_dim
        num_heads = num_heads

        self.pe = PositionalEncoding(embed_dim*2, 0, max_len=32)

        self.transformerEncoderStandardLayer = StandardTransformerEncoderLayer(
            embed_dim*2,
            num_heads, 32, 0,
            torch.nn.functional.relu,
            1e-5, True
        )

        encoder_norm = LayerNorm(embed_dim*2, eps=1e-5)
        self.encoder = TransformerEncoder(
            self.transformerEncoderStandardLayer,
            num_layers=num_layers, norm=encoder_norm
        )

        self.value_key = nn.Conv1d(observation_space['torsion_angles'].shape[1], embed_dim, (1,))
        self.query = nn.Embedding(len(RESIDUE_LETTERS) + 2, embed_dim)

        self._features_dim = 2*embed_dim * observation_space['torsion_angles'].shape[0] + 2

    def forward(self, observations) -> th.Tensor:
        embedded_angle = transpose(observations['torsion_angles'], 1, 2)
        embedded_angle = self.value_key(embedded_angle)
        embedded_angle = transpose(embedded_angle, 1, 2)
        embedded_seq = self.query(observations['amino_acid'].type(torch.IntTensor))
        x = torch.cat((embedded_seq, embedded_angle), dim=2)
        x = self.pe(x)

        x = self.encoder(x)
        x = torch.nn.Flatten()(x)
        x = torch.cat((x, observations['energy'], observations['step']), dim=1)
        return x


class ActorCriticTransformerJointInputPolicy(MaskableActorCriticPolicy):

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        features_extractor_class: Type[BaseFeaturesExtractor] = JointInputFeatureExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(ActorCriticTransformerJointInputPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )