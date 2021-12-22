import gym
import torch
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn, transpose, Tensor
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from torch.nn import LayerNorm

from custom_policies.transformer_encoder_layer import TransformerEncoderLayer, TransformerEncoder
import math

RESIDUE_LETTERS = [
    'R', 'H', 'K',
    'D', 'E',
    'S', 'T', 'N', 'Q',
    'C', 'G', 'P',
    'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'empty'
]

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.transpose(1, 0)
        x = x + self.pe[:x.size(0)]
        return x.transpose(1, 0)

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)
        embed_dim = 16
        num_heads = 2

        self.pe = PositionalEncoding(embed_dim, 0, max_len=32)

        # self.transformerEncoderLayer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=32)
        self.transformerEncoderStandardLayer = torch.nn.TransformerEncoderLayer(2*embed_dim, num_heads, 32, 0,
                                                    torch.nn.functional.relu, 1e-5, True, False,)
        encoder_norm = LayerNorm(2*embed_dim, eps=1e-5, )
        self.encoder = torch.nn.TransformerEncoder(
            self.transformerEncoderStandardLayer,
            num_layers=1, norm=encoder_norm)
        self.value_key = nn.Conv1d(observation_space['torsion_angles'].shape[1], embed_dim, (1,))
        self.query = nn.Embedding(len(RESIDUE_LETTERS) + 2 , embed_dim)

        self._features_dim = 2*embed_dim * observation_space['torsion_angles'].shape[0] + 2

    def forward(self, observations) -> th.Tensor:

        # self.extractors contain nn.Modules that do all the processing.
        embedded_angle = transpose(observations['torsion_angles'], 1, 2)
        embedded_angle = self.value_key(embedded_angle)
        embedded_angle = transpose(embedded_angle, 1, 2)
        embedded_seq = self.query(observations['amino_acid'].type(torch.IntTensor))
        embedded_angle = self.pe(embedded_angle)
        embedded_seq = self.pe(embedded_seq)
        stacked = torch.cat((embedded_angle,embedded_seq ), dim=2)
        x = self.encoder(stacked)
        x = torch.nn.Flatten()(x)
        x = torch.cat((x, observations['energy'], observations['step']), dim=1)
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return x


class ActorCriticTransformerPolicy(ActorCriticPolicy):

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
        features_extractor_class: Type[BaseFeaturesExtractor] = CustomCombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(ActorCriticTransformerPolicy, self).__init__(
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