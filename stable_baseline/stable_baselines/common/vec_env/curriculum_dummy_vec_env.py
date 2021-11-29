import numpy as np
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import os

try:
    LEVELS = [os.path.join('benchmark', x) for x in np.sort(os.listdir('protein_data/benchmark'))]
except:
    pass

# LEVELS = ['benchmark/bench_1']
class CurriculumDummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        DummyVecEnv.__init__(self, env_fns)
        self.average_progress = []
        self.probes_to_account = 1000
        self.level_generator = (x for x in LEVELS)

    def _append_distance_result(self, best_distance, start_distance):
        norm_distance = best_distance / start_distance
        if len(self.average_progress) < self.probes_to_account:
            self.average_progress.append(norm_distance)
        else:
            self.average_progress.append(norm_distance)
            self.average_progress.pop(0)

    def _increase_level(self):
        level_folder = next(self.level_generator)
        for env_idx in range(self.num_envs):
            self.envs[env_idx].set_level(level_folder)
        self.average_progress = []

    def reset(self):
        self._increase_level()
        masks = []
        for env_idx in range(self.num_envs):
            obs, mask = self.envs[env_idx].reset()
            masks.append(mask)

            self._save_obs(env_idx, obs)
        return self._obs_from_buf(), masks

    def step_wait(self):
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], info = \
                self.envs[env_idx].step(self.actions[env_idx])
            if self.buf_dones[env_idx]:
                self._append_distance_result(info["best"], info["start"])
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]['terminal_observation'] = obs
                if len(self.average_progress) == self.probes_to_account and np.mean(self.average_progress) < 0.3:
                    print("Next Level")
                    self._increase_level()
                obs, mask = self.envs[env_idx].reset()
                info["action_mask"] = mask
            self._save_obs(env_idx, obs)
            self.buf_infos[env_idx] = info
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())
