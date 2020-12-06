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

    model = DictPPO2(TransformerPolicy, env, verbose=1, tensorboard_log='./log', n_steps=8, ent_coef=0.001, noptepochs=5,
                 nminibatches=8, full_tensorboard_log=False, learning_rate=1e-4, cliprange=0.2, cliprange_vf=-1, with_action_mask=True)
    model.learn(total_timesteps=1000000)
    model.save('model')
    # model.load('model.zip')

if __name__ == '__main__':
    run()