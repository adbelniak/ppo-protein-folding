import os
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.utils import safe_mean


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

    def _add_metric(self, info):
        reward = info.get("episode")['r']
        self.metric_buffer.append(reward)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            if done:
                self._add_metric(info)

        mean_metric_value = safe_mean(self.metric_buffer)
        should_save_model = mean_metric_value > self.best_threshold
        is_not_too_often = self.num_timesteps > self.min_step_freq + self.last_step_update

        if self.min_step < self.num_timesteps and should_save_model and is_not_too_often:
            path = os.path.join(self.model.logger.dir, f"{self.best_model_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)

            print(f"Saving model checkpoint to {path}")
            self.best_threshold = mean_metric_value
            self.last_step_update = self.num_timesteps

        return True


class SaveOnBestDistance(SaveBestCallback):

    def _add_metric(self, info):
        distance = info['best'] / info['start']
        self.metric_buffer.append(distance)
