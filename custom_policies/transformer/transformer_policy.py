import tensorflow as tf
from gym_rosetta.envs.protein_fold_env import RESIDUE_LETTERS

from stable_baselines.common.policies import ActorCriticPolicy, mlp_extractor, RecurrentActorCriticPolicy
from stable_baselines.common.tf_layers import linear, lstm
from .transformer import Encoder
import numpy as np

class TransformerPolicy(ActorCriticPolicy):
    _encoder = None
    _residue_encoder = None

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=True, layers=None,
                 net_arch=None,
                 act_fun=tf.nn.relu, feature_extraction="mlp", with_action_mask=False, **kwargs):
        super(TransformerPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, with_action_mask=with_action_mask)
        encoder = TransformerPolicy.get_common_police_network()
        residue_encoder = TransformerPolicy.get_residue_encoder_networt()


        self._kwargs_check(feature_extraction, kwargs)

        if net_arch is None:
            if layers is None:
                layers = [128, ]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=reuse):
            transformer_encoded = encoder(self.processed_obs['backbone'], None)
            transformer_amino = residue_encoder(self.processed_obs['amino_acid'], None)
            # matmul_qk = tf.matmul(transformer_amino, transformer_encoded, transpose_a=True)
            # with_energy = tf.layers.flatten(transformer_encoded)
            # amino = tf.layers.flatten(transformer_amino)
            # encode = transformer_encoded(self.processed_obs['backbone'])
            acid_code = tf.layers.Dense(8)(tf.layers.flatten(transformer_amino))
            with_energy = tf.keras.layers.Concatenate()(
                [acid_code, tf.layers.flatten(transformer_encoded), self.processed_obs['current_distance']])
            pi_latent, vf_latent = mlp_extractor(with_energy, net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False, action_mask=None):
        feed_dict = {self.obs_ph[key]: obs[key] for key in obs.keys()}
        if action_mask is not None and len(action_mask) != 0:
            feed_dict[self.action_mask_ph] = np.array(action_mask, dtype=np.float32)
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   feed_dict)
        else:
            action, value, neglogp, proba = self.sess.run([self.action, self.value_flat, self.neglogp, self.policy_proba],
                                                   feed_dict)
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None, action_mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        feed_dict = {self.obs_ph[key]: obs[key] for key in obs.keys()}
        return self.sess.run(self.value_flat, feed_dict)

    @classmethod
    def get_lstm(cls):
        if not cls._encoder:
            cls._encoder = tf.keras.layers.LSTM(64)
        return cls._encoder
    @classmethod
    def get_common_police_network(cls):
        num_layers = 2
        d_model = 2
        num_heads = 2
        dff = 32
        pe_input = 32
        rate = 0.1

        if not cls._encoder:
            cls._encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate, extended_capacity=4)
        return cls._encoder

    @classmethod
    def get_residue_encoder_networt(cls):
        num_layers = 2
        d_model = len(RESIDUE_LETTERS)
        num_heads = 3
        dff = 64
        pe_input = 32
        rate = 0.1

        if not cls._residue_encoder:
            cls._residue_encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)
        return cls._residue_encoder



class TransformerLSTMPolicy(RecurrentActorCriticPolicy):
    _encoder = None
    _residue_encoder = None

    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, layers=None,
                 net_arch=None, act_fun=tf.tanh, layer_norm=False, feature_extraction="cnn", with_action_mask=False,
                 **kwargs):
        super(TransformerLSTMPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                         state_shape=(2 * n_lstm,), reuse=reuse,
                                         scale=(feature_extraction == "mlp"), with_action_mask=with_action_mask)
        encoder = TransformerPolicy.get_common_police_network()
        residue_encoder = TransformerPolicy.get_residue_encoder_networt()


        self._kwargs_check(feature_extraction, kwargs)

        if net_arch is None:
            if layers is None:
                layers = [128, ]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=reuse):
            # transformer_encoded = encoder(self.processed_obs['backbone'], None)
            # transformer_amino = residue_encoder(self.processed_obs['amino_acid'], None)
            # matmul_qk = tf.matmul(transformer_amino, transformer_encoded, transpose_a=True)
            # with_energy = tf.layers.flatten(transformer_encoded)
            # amino = tf.layers.flatten(transformer_amino)

            with_energy = tf.keras.layers.Concatenate()(
                [tf.layers.flatten(self.processed_obs['backbone']), self.processed_obs['current_distance']])
            pi_latent, vf_latent = mlp_extractor(with_energy, net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False, action_mask=None):
        feed_dict = {self.obs_ph[key]: obs[key] for key in obs.keys()}
        if action_mask is not None and len(action_mask) != 0:
            feed_dict[self.action_mask_ph] = np.array(action_mask, dtype=np.float32)
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   feed_dict)
        else:
            action, value, neglogp, proba = self.sess.run([self.action, self.value_flat, self.neglogp, self.policy_proba],
                                                   feed_dict)
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None, action_mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        feed_dict = {self.obs_ph[key]: obs[key] for key in obs.keys()}
        return self.sess.run(self.value_flat, feed_dict)

    @classmethod
    def get_common_police_network(cls):
        num_layers = 2
        d_model = 3
        num_heads = 3
        dff = 128
        pe_input = 16
        rate = 0.1

        if not cls._encoder:
            cls._encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)
        return cls._encoder


    @classmethod
    def get_residue_encoder_networt(cls):
        num_layers = 2
        d_model = len(RESIDUE_LETTERS)
        num_heads = 3
        dff = 128
        pe_input = 16
        rate = 0.1

        if not cls._residue_encoder:
            cls._residue_encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)
        return cls._residue_encoder
