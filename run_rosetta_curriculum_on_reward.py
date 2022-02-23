from stable_baselines3 import MaskablePPO
import argparse
from collections import deque

import gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from custom_policies.transformer_joint_input_encoder_pytorch import \
    ActorCriticTransformerJointInputPolicy
from custom_policies.transformer_maskable_pytorch import MaskableActorCriticTransformerPolicy
from save_callbacks import SaveBestCallback, SaveOnBestDistance, CurriculumCallback, \
    CurriculumScrambleCallback, CurriculumScrambleCallbackOnReward
from custom_policies.transformer_multiple_layes_pytorch import ActorCriticTransformerMultipleLayersPolicy


def make_env(env_id, rank, seed=0, **kwargs):
    def _init():
        env = gym.make(env_id, **kwargs)
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

def environment_settings():
    settings = {
        "distance_reward_weight": 0.0,
        'goal_delta': 0.2
    }
    return settings

if __name__ == '__main__':
    args = arg_parse()
    settings = environment_settings()

    env = DummyVecEnv([make_env('gym_rosetta:protein-fold-v0', i, **settings) for i in range(256)])
    n_timesteps = 15000000
    policy_kwargs = {
        "features_extractor_kwargs": {
            "embedding_dim": 16,
            "num_heads": 2
        }
    }
    save_on_reward = SaveBestCallback(window_size=500, min_step=500000, min_step_freq=1000)
    save_on_distance = SaveOnBestDistance(window_size=500, min_step=1000000, min_step_freq=1000, best_model_prefix='best_distance_model')
    curriculum_calback = CurriculumScrambleCallbackOnReward(
        threshold_delta=3.5, step_distance_level=0.05, window_size=45000,
        envs=env, min_step=100000, start_value=0.75, not_to_early_threshold=60000, step_to_increase=2000000
    )

    single_process_model = MaskablePPO(ActorCriticTransformerJointInputPolicy, env,  verbose=1, clip_range=0.1,
                               tensorboard_log='./logs',  n_steps=32, ent_coef=0.001, policy_kwargs=policy_kwargs)

    single_process_model.learn(n_timesteps, callback=[TensorboardCallback(20), save_on_reward, save_on_distance, curriculum_calback])
    single_process_model.save(args.saving_directory)
