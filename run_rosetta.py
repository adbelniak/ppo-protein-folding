from collections import deque

import gym
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import time
from custom_policies.transformer_pytorch import ActorCriticTransformerPolicy


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, log_interval, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.distance_buffer = deque(maxlen=log_interval)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            if done:
                self.logger.record_mean('protein_distance/{}'.format(info['name']), info['best'])
                self.logger.record_mean('protein_distance_mean', info['best'] / info['start'])

                # self.logger.dump(step=self.num_timesteps)
        return True

if __name__ == '__main__':

    env = DummyVecEnv([make_env('gym_rosetta:protein-fold-v0', i) for i in range(16)])
    # # env = gym.make('gym_rosetta:protein-fold-v0')
    n_timesteps = 1000000

    single_process_model = PPO(ActorCriticTransformerPolicy, env,  verbose=1,
                               tensorboard_log='./logs',  n_steps=16, ent_coef=0.001)

    start_time = time.time()
    single_process_model.learn(n_timesteps, callback=TensorboardCallback(20))
    total_time_single = time.time() - start_time