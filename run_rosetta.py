import gym
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import time
from gym_rosetta.envs.protein_fold_env import ProteinFoldEnv

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

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            if done:
                self.logger.record('protein_distance/{}'.format(info['name']), info['best'])
                # self.logger.dump(step=self.num_timesteps)
        return True

if __name__ == '__main__':

    # env = DummyVecEnv([make_env('gym_rosetta:protein-fold-v0', i) for i in range(16)])
    # # env = gym.make('gym_rosetta:protein-fold-v0')
    #
    # model = A2C('MultiInputPolicy', env, verbose=1, tensorboard_log='./logs', n_steps=50)
    #
    # # model.learn(total_timesteps=10000, callback=TensorboardCallback())
    #
    n_timesteps = 300000
    #
    # # Multiprocessed RL Training
    # start_time = time.time()
    # model.learn(n_timesteps)
    # total_time_multi = time.time() - start_time
    #
    # print(
    #     f"Took {total_time_multi:.2f}s for multiprocessed version - {n_timesteps / total_time_multi:.2f} FPS")

    # Single Process RL Training
    # env = gym.make('gym_rosetta:protein-fold-v0')
    # env = DummyVecEnv([ProteinFoldEnv])
    single_process_model = DQN('MultiInputPolicy', 'gym_rosetta:dqn-protein-fold-v0',gradient_steps=5, learning_starts=50000, train_freq=15, verbose=1, tensorboard_log='./logs', exploration_fraction=0.3, buffer_size=500000)
    # single_process_model = A2C('MultiInputPolicy', 'gym_rosetta:protein-fold-v0',  verbose=1, tensorboard_log='./logs',  n_steps=15, )

    start_time = time.time()
    single_process_model.learn(n_timesteps, callback=TensorboardCallback())
    total_time_single = time.time() - start_time

    # print(
    #     f"Took {total_time_single:.2f}s for single process version - {n_timesteps / total_time_single:.2f} FPS")
    #
    # print(
    #     "Multiprocessed training is {:.2f}x faster!".format(total_time_single / total_time_multi))