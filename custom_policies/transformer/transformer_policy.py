import tensorflow as tf

from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn, mlp_extractor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm
import numpy as np
from .transformer import Transformer, Encoder


class TransformerPolicy(ActorCriticPolicy):
    _encoder = None

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=True, layers=None,
                 net_arch=None,
                 act_fun=tf.nn.relu, feature_extraction="mlp", **kwargs):
        super(TransformerPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse)

        num_layers = 2
        d_model = 2
        num_heads = 2
        dff = 128
        pe_input = 64
        rate = 0.1
        # sample_transformer = Transformer(
        #     num_layers=2, d_model=3, num_heads=3, dff=2048,
        #     input_vocab_size=8500, target_vocab_size=8000,
        #     pe_input=10000, pe_target=6000)

        encoder = TransformerPolicy.get_common_police_network()
        # extracted_features = tf.keras.layers.Dense(128, activation='relu')(self.processed_obs)
        # extracted_features = tf.keras.layers.MaxPooling1D(pool_size=2)(extracted_features)
        # extracted_features = tf.keras.layers.Conv1D(128, kernel_size=3, padding='same')(extracted_features)
        # extracted_features = tf.keras.layers.MaxPooling1D(pool_size=2)(extracted_features)

        # lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=128)
        # extracted_features = nature_cnn(self.processed_obs, **kwargs)

        self._kwargs_check(feature_extraction, kwargs)

        if net_arch is None:
            if layers is None:
                layers = [128, ]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=reuse):
            # x_image = tf.keras.layers.Reshape((-1, 3, 1))(self.processed_obs['residue_chain'])  # batch_size  x board_x x board_y x 1

            transformer_encoded = encoder(self.processed_obs['backbone'], None)

            with_energy = tf.layers.flatten(transformer_encoded)
            with_energy = tf.keras.layers.Concatenate()(
                [with_energy, self.processed_obs['current_residue_pose'], self.processed_obs['residue_number'],
                 self.processed_obs['step_to_end']])
            pi_latent, vf_latent = mlp_extractor(with_energy, net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):

        feed_dict = {self.obs_ph[key]: obs[key] for key in obs.keys()}

        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   feed_dict)
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   feed_dict)
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        feed_dict = {self.obs_ph[key]: obs[key] for key in obs.keys()}
        return self.sess.run(self.value_flat, feed_dict)

    @classmethod
    def get_common_police_network(cls):
        num_layers = 2
        d_model = 2
        num_heads = 2
        dff = 128
        pe_input = 64
        rate = 0.1
        # sample_transformer = Transformer(
        #     num_layers=2, d_model=3, num_heads=3, dff=2048,
        #     input_vocab_size=8500, target_vocab_size=8000,
        #     pe_input=10000, pe_target=6000)

        if not cls._encoder:
            cls._encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)
        return cls._encoder
