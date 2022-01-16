from stable_baselines3 import MaskablePPO
import argparse
from collections import deque

import gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from custom_policies.transformer_maskable_pytorch import MaskableActorCriticTransformerPolicy
from save_callbacks import SaveBestCallback, SaveOnBestDistance


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
                # self.logger.record_mean('protein_energy/{}'.format(info['name']), info['best_energy'])

                self.logger.record_mean('protein_distance_mean', info['best'] / info['start'])
                self.logger.record_mean('protein_energy_mean', info['best_energy'])
                self.logger.record_mean('last_step/protein_distance_mean', info['final_distance'] / info['start'])
                # self.logger.dump(step=self.num_timesteps)
        return True


def arg_parse():
    parser = argparse.ArgumentParser(description='RL training')
    parser.add_argument('--saving_directory', dest='saving_directory', action='store', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    env = DummyVecEnv([make_env('gym_rosetta:protein-fold-v0', i) for i in range(256)])
    n_timesteps = 10000000
    policy_kwargs = {
        "features_extractor_kwargs": {
            "embedding_dim": 16,
            "num_heads": 2
        }
    }
    save_on_reward = SaveBestCallback(window_size=100, min_step=300000, min_step_freq=1000)
    save_on_distance = SaveOnBestDistance(window_size=100, min_step=500000, min_step_freq=1000, best_model_prefix='best_distance_model')

    single_process_model = MaskablePPO(MaskableActorCriticTransformerPolicy, env,  verbose=1,
                               tensorboard_log='./logs',  n_steps=16, ent_coef=0.001, policy_kwargs=policy_kwargs)

    single_process_model.learn(n_timesteps, callback=[TensorboardCallback(20), save_on_reward, save_on_distance])
    single_process_model.save(args.saving_directory)
