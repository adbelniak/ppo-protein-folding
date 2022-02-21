import os
import re
from collections import deque

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import DummyVecEnv


class SaveBestCallback(BaseCallback):
    """
    Save best model based on highest window mean reward. Window size is 100
    """

    def __init__(self, window_size=100, min_step=300000, min_step_freq=1000, best_model_prefix='best_model', verbose=0):
        super(SaveBestCallback, self).__init__(verbose)
        self.metric_buffer = deque(maxlen=window_size)
        self.best_threshold = - 1000
        self.min_step = min_step
        self.best_model_prefix = best_model_prefix
        self.last_step_update = 0
        self.min_step_freq = min_step_freq
        self.comparison_fun = lambda a, b: a > b
        self.best_df = []

    def _add_metric(self, info):
        reward = info.get("episode")['r']
        self.metric_buffer.append(reward)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            if done:
                self._add_metric(info)

        mean_metric_value = safe_mean(self.metric_buffer)
        should_save_model = self.comparison_fun(mean_metric_value, self.best_threshold)
        is_not_too_often = self.num_timesteps > self.min_step_freq + self.last_step_update

        if self.min_step < self.num_timesteps and should_save_model and is_not_too_often:
            path = os.path.join(self.model.logger.dir, f"{self.best_model_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)

            print(f"Saving model checkpoint to {path}")
            self.best_threshold = mean_metric_value
            self.last_step_update = self.num_timesteps
            self.best_df.append({"best_threshold": self.best_threshold, "num_timesteps": self.num_timesteps})
        return True

    def _on_training_end(self):
        df = pd.DataFrame(self.best_df)
        path = os.path.join(self.model.logger.dir,
                            f"{self.best_model_prefix}_description.csv")
        df.to_csv(path)


class SaveOnBestDistance(SaveBestCallback):
    def __init__(self, **kwargs):
        super(SaveOnBestDistance, self).__init__(**kwargs)
        self.best_threshold = 2
        self.comparison_fun = lambda a, b: a < b

    def _add_metric(self, info):
        distance = info['best'] / info['start']
        self.metric_buffer.append(distance)


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


class CurriculumCallback(BaseCallback):
    def __init__(
        self,
        threshold_increase: float = 0.3,
        window_size:int = 500,
        envs: DummyVecEnv = None,
        verbose: int = 0,
        min_step: int = 10000
    ):
        super(CurriculumCallback, self).__init__(verbose=verbose)
        self.average_progress = deque(maxlen=window_size)
        self.probes_to_account = window_size
        self.init_level_generator()
        self.dummyVecEnv = envs
        self.threshold_increase = threshold_increase
        self.best_df = []
        self.current_level = None
        self.min_step=min_step
        self.best_model_prefix = 'curriculum_distance'
        self._increase_level(False)

    def init_level_generator(self):
        dirs = os.listdir('protein_data/benchmark')
        dirs = [directory for directory in dirs if os.path.isdir(os.path.join('protein_data/benchmark', directory))]
        dirs.sort(key=natural_keys)
        levels = [os.path.join('benchmark', x) for x in dirs]
        self.level_generator = (x for x in levels)

    def _increase_level(self, save_model=True):
        level_folder = next(self.level_generator)
        self.current_level = level_folder
        for env in self.dummyVecEnv.envs:
            env.set_level(level_folder)
        self.average_progress = deque(maxlen=self.probes_to_account)
        if save_model:
            path = os.path.join(self.model.logger.dir,
                                f"{self.best_model_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            df = pd.DataFrame(self.best_df)
            path = os.path.join(self.model.logger.dir,
                                f"{self.best_model_prefix}_description.csv")
            df.to_csv(path)

    def _add_metric(self, info):
        distance = info['best'] / info['start']
        self.average_progress.append(distance)

    def _on_step(self) -> bool:
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            if done:
                self._add_metric(info)

        not_too_early = self.min_step < self.num_timesteps
        filled_buffer = len(self.average_progress) >= self.probes_to_account

        if not_too_early and filled_buffer and np.mean(self.average_progress) < self.threshold_increase:
            self._increase_level()
            self.dummyVecEnv.reset()
            print("Next Level: {}".format(self.current_level))

            self.best_df.append({"level": self.current_level, "num_timesteps": self.num_timesteps})

        return True


class CurriculumDistanceCallback(CurriculumCallback):
    def __init__(
        self,
        threshold_delta: float = 0.1,
        step_distance_level: float = 0.05,
        **kwargs
    ):
        self.step_distance_level = step_distance_level
        super(CurriculumDistanceCallback, self).__init__(**kwargs)
        self.threshold_delta = threshold_delta
        self.best_model_prefix = 'curriculum_distance_reduction'
        self.step_in_level = 0
        self.step_to_increase = 500000

    def init_level_generator(self):
        levels = np.arange(0.9, 0.05, -self.step_distance_level)
        self.level_generator = (x for x in levels)

    def _increase_level(self, save_model=True):
        try:
            level_folder = next(self.level_generator)
            self.current_level = level_folder
        except:
            return False

        for env in self.dummyVecEnv.envs:
            env.set_level_delta_goal(self.current_level)
        self.average_progress = deque(maxlen=self.probes_to_account)
        self.best_df.append({"level": self.current_level, "num_timesteps": self.num_timesteps})
        self.step_in_level = 0
        if save_model:
            path = os.path.join(self.model.logger.dir,
                                f"{self.best_model_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            df = pd.DataFrame(self.best_df)
            path = os.path.join(self.model.logger.dir,
                                f"{self.best_model_prefix}_description.csv")
            df.to_csv(path)
        return True

    def _add_metric(self, info):
        distance = info['best'] / info['start']
        self.average_progress.append(distance)

    def _on_step(self) -> bool:
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            if done:
                self._add_metric(info)

        self.step_in_level += len(self.locals['infos'])
        not_too_early = self.min_step < self.num_timesteps
        filled_buffer = len(self.average_progress) >= self.probes_to_account
        exceed_level = np.mean(self.average_progress) < (self.current_level + self.threshold_delta)
        force_increase = self.step_to_increase < self.step_in_level
        if (not_too_early and filled_buffer and exceed_level) or force_increase:
            was_increased = self._increase_level()
            if was_increased:
                self.dummyVecEnv.reset()
                print("Next Level: {}".format(self.current_level))
        return True
