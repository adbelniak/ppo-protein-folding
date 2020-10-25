from stable_baseline.stable_baselines.common.cmd_util import make_rosetta_env
from pyrosetta import init
from stable_baselines.common.vec_env.rossetta_vec_normalize import RossettaVecNormalize
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

import gym
from stable_baselines.ppo2.dict_ppo2 import DictPPO2

from custom_policies.custom_policy import LstmCustomPolicy
from custom_policies.transformer.transformer_policy import TransformerPolicy


def run():
    init()

    seed = 2
    env = make_rosetta_env("gym_rosetta:protein-fold-v0", 64, seed, use_subprocess=False)
    # env.shuffle = True
    # env = RossettaVecNormalize(env)

    model = DictPPO2(LstmCustomPolicy, env, verbose=1, tensorboard_log='./log', n_steps=64, ent_coef=0.0005, noptepochs=5,
                 nminibatches=8, full_tensorboard_log=False, learning_rate=1e-4, cliprange=0.1)
    model.learn(total_timesteps=2000000)

if __name__ == '__main__':
    run()