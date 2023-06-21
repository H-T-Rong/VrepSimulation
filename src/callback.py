import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

# model = SAC("MlpPolicy", "Pendulum-v1", tensorboard_log="/tmp/sac/", verbose=1)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.tr = None
        self.ts = None
        self.sigma = None

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        # value = np.random.random()
        # self.logger.record("random_value", value)
        return True

    def _on_rollout_end(self) -> None:
        self.tr, self.ts, self.sigma = self.training_env.env_method('get_indicator')[0]
        self.logger.record("ts", self.ts)
        self.logger.record("tr", self.tr)
        self.logger.record("sigma", self.sigma)
        return super()._on_rollout_end()

# model.learn(50000, callback=TensorboardCallback())