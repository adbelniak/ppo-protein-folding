import argparse
import os
from copy import deepcopy

import gym
import numpy as np
import subprocess

import pandas as pd

from stable_baselines3 import A2C, DQN, PPO, MaskablePPO
from stable_baselines3.common.utils import set_random_seed, safe_mean
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.base_vec_env  import VecEnvObs
from stable_baselines3.common.maskable.utils import get_action_masks, is_masking_supported


class EvalVec(DummyVecEnv):
    """ VecEnv for evaluation. This implementation do not reset env if done. """
    def reset(self, **kwargs) -> VecEnvObs:
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset(**kwargs)
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()

    def step_wait(self):
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            if self.buf_dones[env_idx]:
                self.buf_infos[env_idx]["terminal_observation"] = obs

            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))




def make_env(env_id, rank, dataset_dir, max_move_amount=128, seed=0):
    def _init():
        env = gym.make(env_id, **{"max_move_amount": max_move_amount, "level_dir": dataset_dir})
        # env = Monitor(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


def eval_single_run(model, env, num_steps, protein_name, deterministic=True):
    """Return best 5 distances based on min score for each env run"""
    distance_energy = []
    obs = env.reset(protein_name=protein_name)
    for step in range(num_steps):
        action_masks = get_action_masks(env)
        action, _state = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
        obs, reward, dones, infos = env.step(action)

        for i, info in enumerate(infos):
            norm_distance = info['distance'] / info['start']
            distance_energy.append({"name": protein_name,
                                    "distance": norm_distance,
                                    "energy": info['current_energy'],
                                   "env_id": i})

    stats_df = pd.DataFrame(distance_energy)

    df = stats_df.sort_values(["env_id", "energy"]).groupby("env_id").head(5)
    df['method_type'] = 'energy'

    distance_df = stats_df.sort_values(["env_id", "distance"]).groupby("env_id").head(1)
    distance_df['method_type'] = 'best_distance'
    return pd.concat((distance_df, df), axis=0)

def eval_single_model(dataset_dir, model_path, descript_file, deterministic=True):

    test_df = pd.read_csv(os.path.join(dataset_dir, descript_file))
    episode_len = 128
    print(model_path)
    model = MaskablePPO.load(model_path)
    env = EvalVec([make_env('gym_rosetta:protein-fold-v0', i, dataset_dir, episode_len) for i in range(2)])

    # stats = eval_single_run(model, env, 128, 'cos')
    eval_scores = []
    for i, row in test_df.iterrows():
        series = eval_single_run(model, env, episode_len, row['protein_name'], deterministic=deterministic)
        eval_scores.append(series)
    eval_scores = pd.concat(eval_scores, axis=0)
    print(eval_scores['distance'].mean())
    return eval_scores


def check_if_exists(results_df, current_model_path):
    if results_df is None:
        return False
    return current_model_path in results_df['model_name'].unique()


def save_results(model_results, results_path, previous_scores):
    if previous_scores is not None:
        pd.concat([previous_scores, *model_results], axis=0).to_csv(results_path)
    else:
        pd.concat(model_results, axis=0).to_csv(results_path)


def eval_all_models(model_list, dataset_dir, results_path, descript_file='test.csv', previous_scores=None, deterministic=True):
    model_results = []
    for index, row in model_list.iterrows():
        if check_if_exists(previous_scores, row['model_local_path']):
            continue
        result = eval_single_model(dataset_dir, row['model_local_path'], descript_file, deterministic=deterministic)
        result['experiment'] = row['experiment']
        result['model_name'] = row['model_local_path']
        model_results.append(result)
    save_results(model_results, results_path, previous_scores)


def read_models_description(model_dir, model_description_path: str):
    models_description = pd.read_csv(model_description_path, names=['id', 'model_remote_path', 'experiment'], skiprows=1)
    models_description[['rest', 'experiment', 'model_local_path']] =\
        models_description['model_remote_path'].str.split('/', -1, expand=True)

    models_description['model_local_path'] = models_description\
        .apply((lambda row: os.path.join(model_dir, row['experiment'], row['model_local_path'])), axis=1)
    return models_description.drop(['model_remote_path', 'rest'], axis=1)


def arg_parse():
    parser = argparse.ArgumentParser(description='RL training')
    parser.add_argument('--dataset_path', dest='dataset_path', action='store', type=str, default="../protein_data/baseline")
    parser.add_argument('--results_path', dest='results_path', action='store', type=str,
                        default='../experiment_old_instance_3/reward_shaping_results_34_35.csv')
    parser.add_argument('--deterministic_agent', dest='deterministic_agent', action='store', type=bool, default=True)
    parser.add_argument('--append_to_exists', dest='append_to_exists', action='store', type=bool, default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    test_file = 'test2.csv'
    deterministic = args.deterministic_agent
    previous_scores=None
    # if args.append_to_exists:
    #     previous_scores = pd.read_csv(args.results_path)
    # eval_test_set(args.dataset_path, test_file, args.model_path)
    model_list = read_models_description('../experiment_old_instance_3', '../experiment_old_instance_3/filtered_models_reward_shaping.csv')
    eval_all_models(model_list, args.dataset_path, args.results_path, test_file,
                    previous_scores=previous_scores, deterministic=deterministic)
    print(model_list)