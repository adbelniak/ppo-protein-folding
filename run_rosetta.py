from stable_baseline.stable_baselines.common.cmd_util import make_rosetta_env
from pyrosetta import init
from stable_baselines.common.vec_env.rossetta_vec_normalize import RossettaVecNormalize
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

import gym
from stable_baselines.ppo2.dict_ppo2 import DictPPO2

from custom_policies.transformer.transformer_policy import TransformerPolicy


def run():
    init()

    seed = 0
    # env = gym.make('gym_rosetta:protein-fold-v0')
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    env = make_rosetta_env("gym_rosetta:protein-fold-v0", 128, seed, use_subprocess=False)
    # env.shuffle = True
    env = RossettaVecNormalize(env)

    model = DictPPO2(TransformerPolicy, env, verbose=1, tensorboard_log='./log', n_steps=32, ent_coef=0.001, noptepochs=5,
                 nminibatches=8, full_tensorboard_log=True, learning_rate=1e-3)
    model.learn(total_timesteps=1000000)

if __name__ == '__main__':
    run()