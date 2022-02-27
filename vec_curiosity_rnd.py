import logging
import os
from typing import List, Dict

import numpy as np
import tensorflow as tf
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import obs_as_tensor, get_device
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from stable_baselines3.common.vec_env.util import obs_space_info
import torch.nn.functional as F
from torch.nn import init

from custom_policies.transformer_encoder_layer import TransformerEncoderLayer

from custom_policies.transformer_maskable_pytorch import PositionalEncoding, RESIDUE_LETTERS
import gym
import torch
import torch as th
from torch import nn, transpose, Tensor
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import torch.optim as optim


class RNDModel(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict=None, embedding_dim=16, num_heads=2):
        super(RNDModel, self).__init__(observation_space, features_dim=1)
        embed_dim = embedding_dim
        num_heads = num_heads

        self.pe = PositionalEncoding(embed_dim, 0, max_len=32)
        self.encoder = TransformerEncoderLayer(embed_dim, num_heads,
                                               dim_feedforward=64)

        self.value_key = nn.Conv1d(observation_space['torsion_angles'].shape[1], embed_dim, (1,))
        self.query = nn.Embedding(len(RESIDUE_LETTERS) + 2, embed_dim)
        self._features_dim = embed_dim * observation_space['torsion_angles'].shape[0] + 2

        self.linear = nn.Linear(self._features_dim, 32)

        for p in self.modules():
            if isinstance(p, nn.Conv1d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, observations) -> th.Tensor:
        embedded_angle = transpose(observations['torsion_angles'], 1, 2)
        embedded_angle = self.value_key(embedded_angle)
        embedded_angle = transpose(embedded_angle, 1, 2)
        embedded_seq = self.query(observations['amino_acid'].type(torch.IntTensor))
        embedded_angle = self.pe(embedded_angle)
        embedded_seq = self.pe(embedded_seq)
        x = self.encoder(embedded_angle, embedded_seq, embedded_angle)
        x = torch.nn.Flatten()(x)
        x = torch.cat((x, observations['energy'], observations['step']), dim=1)

        x = self.linear(x)
        return x

class RNDModelPredictor(RNDModel):
    def __init__(self, **kwargs):
        super(RNDModelPredictor, self).__init__(**kwargs)

        self.linear_predictor = nn.Sequential(
            nn.Linear(self._features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,32)
        )
        for p in self.modules():
            if isinstance(p, nn.Conv1d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()


    def forward(self, observations) -> th.Tensor:
        embedded_angle = transpose(observations['torsion_angles'], 1, 2)
        embedded_angle = self.value_key(embedded_angle)
        embedded_angle = transpose(embedded_angle, 1, 2)
        embedded_seq = self.query(observations['amino_acid'].type(torch.IntTensor))
        embedded_angle = self.pe(embedded_angle)
        embedded_seq = self.pe(embedded_seq)
        x = self.encoder(embedded_angle, embedded_seq, embedded_angle)
        x = torch.nn.Flatten()(x)
        x = torch.cat((x, observations['energy'], observations['step']), dim=1)
        x = self.linear_predictor(x)
        return x


class CuriosityWrapper(DummyVecEnv):
    """
    Random Network Distillation (RND) curiosity reward.
    https://arxiv.org/abs/1810.12894

    :param env: (gym.Env) Environment to wrap.
    :param network: (str) Network type. Can be a "cnn" or a "mlp".
    :param intrinsic_reward_weight: (float) Weight for the intrinsic reward.
    :param buffer_size: (int) Size of the replay buffer for predictor training.
    :param train_freq: (int) Frequency of predictor training in steps.
    :param gradient_steps: (int) Number of optimization epochs.
    :param batch_size: (int) Number of samples to draw from the replay buffer per optimization epoch.
    :param learning_starts: (int) Number of steps to wait before training the predictor for the first time.
    :param filter_end_of_episode: (bool) Weather or not to filter end of episode signals (dones).
    :param filter_reward: (bool) Weather or not to filter extrinsic reward from the environment.
    :param norm_obs: (bool) Weather or not to normalize and clip obs for the target/predictor network. Note that obs returned will be unaffected.
    :param norm_ext_reward: (bool) Weather or not to normalize extrinsic reward.
    :param gamma: (float) Reward discount factor for intrinsic reward normalization.
    :param learning_rate: (float) Learning rate for the Adam optimizer of the predictor network.
    """

    def __init__(self, env_fns, network: str = "mlp", intrinsic_reward_weight: float = 1.0, buffer_size: int = 2*65536,
                 train_freq: int = 16384, gradient_steps: int = 4,
                 batch_size: int = 4096, learning_starts: int = 100, filter_end_of_episode: bool = True,
                 filter_reward: bool = False, norm_obs: bool = False,
                 norm_ext_reward: bool = False, gamma: float = 0.99, learning_rate: float = 0.0001,
                 training: bool = True, _init_setup_model=True):

        DummyVecEnv.__init__(self, env_fns)

        self.network_type = network
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.filter_end_of_episode = filter_end_of_episode
        self.filter_extrinsic_reward = filter_reward
        self.clip_obs = 5
        self.norm_obs = norm_obs
        self.norm_ext_reward = norm_ext_reward
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.training = training

        self.epsilon = 1e-8
        self.int_rwd_rms = RunningMeanStd(shape=(), epsilon=self.epsilon)
        self.ext_rwd_rms = RunningMeanStd(shape=(), epsilon=self.epsilon)
        self.int_ret = np.zeros(self.num_envs)  # discounted return for intrinsic reward
        self.ext_ret = np.zeros(self.num_envs)  # discounted return for extrinsic reward
        env = self.envs[0]
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.obs_rms = {key: RunningMeanStd(shape=obs_space[key].shape, epsilon=self.epsilon) for key in self.keys}

        self.last_obs = None

        self.updates = 0
        self.steps = 0
        # probably not important
        self.last_action = None
        # probably not important
        self.last_obs = None
        self.last_update = 0
        self.probes_to_account = 4000
        self.graph = None
        self.sess = None
        self.observation_ph = None
        self.processed_obs = None
        self.predictor_network = None
        self.target_network = None
        self.params = None
        self.int_reward = None
        self.aux_loss = None
        self.optimizer = None
        self.training_op = None
        self.predictor_mse_loss = nn.MSELoss(reduction='none')
        self.buffer = DictReplayBuffer(buffer_size, env.observation_space, env.action_space, handle_timeout_termination=False)
        self.device = get_device("auto")
        if _init_setup_model:
            self.setup_model()
        self.optimizer = optim.Adam(self.predictor_model.parameters(), lr=0.0001)
        self.observation_space = env.observation_space
        self.logger= None

    def setup_model(self):
        self.graph = tf.Graph()
        env = self.envs[0]
        obs_space = env.observation_space
        self.target_model = RNDModel(obs_space)
        self.predictor_model = RNDModelPredictor(observation_space=obs_space)

    def step_async(self, actions):
        super().step_async(actions)
        self.last_action = actions
        self.steps += self.num_envs


    def get_intrinsic_rewards(self, input_observation):
        input_observation = obs_as_tensor(input_observation, self.device)
        obs = preprocess_obs(input_observation, self.observation_space , False)
        target_value = self.target_model(obs)  # shape: [n,512]
        predictor_value = self.predictor_model(obs)
        intrinsic_reward = (target_value - predictor_value).pow(2).sum(1) / 2
        intrinsic_reward = intrinsic_reward.data.cpu().numpy()
        return intrinsic_reward

    def reset(self):
        batch_obs, masks = [], []
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
            batch_obs.append(obs)
        self.last_obs = batch_obs

        return self._obs_from_buf()

    def step_wait(self) -> VecEnvStepReturn:
        batch_obs = []
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = \
            self.envs[env_idx].step(
                self.actions[env_idx]
            )
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
            self.buffer.add(self.last_obs[env_idx], obs, self.last_action[env_idx],
                            self.buf_rews[env_idx],
                            self.buf_dones[env_idx], self.buf_infos[env_idx])
            batch_obs.append(obs)
        self.last_obs = batch_obs

        obs = self._obs_from_buf()

        if self.training:
            for key in self.obs_rms.keys():
                self.obs_rms[key].update(obs[key])

        loss = self.get_intrinsic_rewards(obs)

        if self.training:
            self._update_ext_reward_rms(np.copy(self.buf_rews))
            self._update_int_reward_rms(loss)

        intrinsic_reward = np.array(loss) / np.sqrt(self.int_rwd_rms.var + self.epsilon)
        print(np.sqrt(self.int_rwd_rms.var + self.epsilon))
        extrinsic_reward = np.copy(self.buf_rews)
        reward = np.squeeze(extrinsic_reward + self.intrinsic_reward_weight * intrinsic_reward)

        self.logger.record_mean("rollout/intr_reward", np.mean(self.intrinsic_reward_weight * intrinsic_reward))
        self.logger.record_mean("rollout/intr_reward_max", np.max(self.intrinsic_reward_weight * intrinsic_reward))
        self.logger.record_mean("rollout/intr_reward_min",
                           np.min(self.intrinsic_reward_weight * intrinsic_reward))

        if self.training and self.steps > self.learning_starts and self.steps - self.last_update > self.train_freq:
            self.updates += 1
            self.last_update = self.steps
            self.learn()
        # self.learn()

        return (self._obs_from_buf(), reward, np.copy(self.buf_dones),
                deepcopy(self.buf_infos))


    def learn(self):
        total_loss = 0
        for _ in range(self.gradient_steps):
            obs_batch, act_batch, rews_batch, next_obs_batch, done_mask = self.buffer.sample(self.batch_size)
            input_observation = obs_as_tensor(obs_batch, self.device)
            obs = preprocess_obs(input_observation, self.observation_space, False)

            target_value = self.target_model(obs)  # shape: [n,512]
            predictor_value = self.predictor_model(obs)
            loss = F.mse_loss(predictor_value, target_value.detach())
            self.optimizer.zero_grad()
            loss.backward()
            total_loss+= loss
            self.optimizer.step()
        self.logger.record_mean("train/predictor_loss",
                                total_loss / self.gradient_steps)
        print("Trained predictor. Avg loss: {}".format(total_loss / self.gradient_steps))

    def _update_int_reward_rms(self, reward: np.ndarray) -> None:
        """Update reward normalization statistics."""
        self.int_ret = self.gamma * self.int_ret + reward
        self.int_rwd_rms.update(self.int_ret)

    def _update_ext_reward_rms(self, reward: np.ndarray) -> None:
        """Update reward normalization statistics."""
        self.ext_ret = self.gamma * self.ext_ret + reward
        self.ext_rwd_rms.update(self.ext_ret)


    def get_parameter_list(self):
        return self.params

    def save(self, save_path):
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # self.saver.save(self.sess, save_path)

        data = {
            'network': self.network_type,
            'intrinsic_reward_weight': self.intrinsic_reward_weight,
            'buffer_size': self.buffer.buffer_size,
            'train_freq': self.train_freq,
            'gradient_steps': self.gradient_steps,
            'batch_size': self.batch_size,
            'learning_starts': self.learning_starts,
            'filter_end_of_episode': self.filter_end_of_episode,
            'filter_extrinsic_reward': self.filter_extrinsic_reward,
            'norm_obs': self.norm_obs,
            'norm_ext_reward': self.norm_ext_reward,
            'gamma': self.gamma,
            'learning_rate': self.learning_rate,
            'int_rwd_rms': self.int_rwd_rms,
            'ext_rwd_rms': self.ext_rwd_rms,
            'obs_rms': self.obs_rms
        }

        params_to_save = self.get_parameters()
        self._save_to_file_zip(save_path, data=data, params=params_to_save)
