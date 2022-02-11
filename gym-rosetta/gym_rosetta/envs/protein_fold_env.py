import json
import os, subprocess, time, signal
from typing import List

import gym
from gym import error, spaces
from gym import utils
from pyrosetta import init, pose_from_pdb, pose_from_sequence
from pyrosetta.rosetta.core.scoring import CA_rmsd
from pyrosetta.rosetta.core.pose import deep_copy

import numpy as np
from angles import normalize
from pyrosetta.teaching import *
ANGLE_MOVE = [-90, -45, -10, -5, 5, 10, 45, 90, -1, 1]
import logging

logger = logging.getLogger(__name__)

RESIDUE_LETTERS = [
    'R', 'H', 'K',
    'D', 'E',
    'S', 'T', 'N', 'Q',
    'C', 'G', 'P',
    'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'empty'
]

MAX_LENGTH = 16

class ProteinLibrary:
    def __init__(self):
        self.library ={}

    def get_protein(self, name):
        if name in self.library:
            return self.library[name].clone()
        self.library[name] = pose_from_pdb(name)
        return self.library[name].clone()

class ProteinFoldEnv(gym.Env, utils.EzPickle):

    _library = ProteinLibrary()

    def __init__(
        self,
        max_move_amount=64,
        level_dir=None,
        distance_reward_weight=0.0,
        abs_distance_reward_weight=0.0,
        angle_reward_weight=0.0,
        energy_reward_weight=0.0
    ):
        super(ProteinFoldEnv, self).__init__()
        self.reward = 0.0
        self.prev_ca_rmsd = None
        self.protein_pose = None
        self._configure_environment()
        self.is_finished = False
        self.move_counter = 0
        self.max_move_amount = max_move_amount
        self.name = None
        self.best_conformations_dict = {}

        self.observation_space = spaces.Dict({
            "amino_acid": spaces.Box(low=0, high=len(RESIDUE_LETTERS), shape=(MAX_LENGTH,), dtype=np.int),
            "torsion_angles": spaces.Box(low=-1, high=1, shape=(MAX_LENGTH, 4,)),
            "energy": spaces.Box(low=-10, high=10, shape=(1,)),
            "step": spaces.Box(low=0, high=1, shape=(1,)),
        })
        self.action_space = spaces.MultiDiscrete([MAX_LENGTH - 1, 2, len(ANGLE_MOVE)])
        self.epoch_counter = 0
        self.encoded_residue_sequence = None
        self.scorefxn = get_fa_scorefxn()
        self.best_distance = np.inf
        self.validation = False
        self.best_conformations_dict = {}
        self.start_distance = 1000
        self.residue_mask = None
        self.shuffle = False
        self.achieved_goal_counter = 0

        self.best_energy = None
        self.prev_energy = None
        self.start_energy = 10000
        self.level_dir = level_dir or 'protein_data/baseline'
        self.offset = 0
        #input due to residue start from index 1 and first residue do not affect on distance
        self.input_shift = 2
        self.distance_reward_weight = distance_reward_weight
        self.abs_distance_reward_weight = abs_distance_reward_weight
        self.angle_reward_weight = angle_reward_weight
        self.energy_reward_weight = energy_reward_weight

    def _configure_environment(self):
        """
        Provides a chance for subclasses to override this method and supply
        a different server configuration. By default, we initialize one
        offense agent against no defenders.
        """
        self._start_rossetta()

    def _start_rossetta(self):
        init(extra_options="-mute all")

    def _get_distance(self, coordinates, target):
        return CA_rmsd(coordinates, target)

    def _move_on_torsion(self, residue_number, move_pose, get_angle, set_angle):
        new_torsion_position = normalize(get_angle(residue_number) + move_pose, -180, 180)
        set_angle(residue_number, new_torsion_position)

    def _move(self, action):
        #rosetta start count from 1, but first residue do not affect on distance
        self.move_counter += 1
        residue_number = action[0] +  self.input_shift
        torsion_number = action[1]
        move_pose_index = action[2]
        move_pose = ANGLE_MOVE[move_pose_index]

        # bez < bo wpada w lokalne minimum - nie wplywa na distance
        if residue_number < self.total_current_residue:
            if torsion_number == 0:
                self._move_on_torsion(residue_number, move_pose, self.protein_pose.phi,
                                      self.protein_pose.set_phi)
            if torsion_number == 1:
                self._move_on_torsion(residue_number, move_pose, self.protein_pose.psi,
                                      self.protein_pose.set_psi)
            return 0.0
        else:
            return -0.2

    def _get_residue_metric(self, resiude_pose, target_residue_pose):
        angle_distance = np.arctan2(np.sin(resiude_pose - target_residue_pose),
                                    np.cos(resiude_pose - target_residue_pose))
        return abs(angle_distance)

    def _encode_residues(self, sequence):
        encoded_residues = np.zeros(MAX_LENGTH)
        # for zero in range(self.offset):
        #     temp = np.zeros(len(RESIDUE_LETTERS))
        #     temp[len(RESIDUE_LETTERS) - 1] = -1
        for i, res in enumerate(sequence):
            encoded_residues[i] = RESIDUE_LETTERS.index(res)
        for zero in range(len(sequence), MAX_LENGTH):
            encoded_residues[zero] = len(RESIDUE_LETTERS)
        return encoded_residues

    def _convert_to_sin_cos(self, angles):
        return np.column_stack((np.cos(np.radians(angles)), np.sin(np.radians(angles))))

    def _get_state(self):
        psis = self._convert_to_sin_cos(
            [self.protein_pose.psi(i + 1) for i in range(self.total_current_residue)])
        phis = self._convert_to_sin_cos([self.protein_pose.phi(i + 1) for i in
                                         range(self.total_current_residue)])

        ca_coordinate = np.divide([self.protein_pose.residue(i + 1).xyz("CA") for i in
                                   range(self.total_current_residue)], 100.0)

        rest_zeros = MAX_LENGTH - self.total_current_residue
        psis = np.concatenate((psis, np.zeros((rest_zeros, 2))))
        phis = np.concatenate((phis, np.zeros((rest_zeros, 2))))
        cord = np.concatenate((ca_coordinate, np.zeros((rest_zeros, 3))))

        # self.prev_energy =

        torsion_angles = np.column_stack((psis, phis))

        return {
            "torsion_angles": torsion_angles,
            # "ca_geometry": padded_ca_coordinate,
            "energy": [self.scorefxn(self.protein_pose) / self.start_energy],
            "amino_acid": self.encoded_residue_sequence,
            "step": self.move_counter / self.max_move_amount
        }

    def save_best_matches(self):
        if self.validation:
            with open('data_valid.json', 'w') as fp:
                json.dump(self.best_conformations_dict, fp)
        else:
            with open('data.json', 'w') as fp:
                json.dump(self.best_conformations_dict, fp)

    def create_residue_mask(self, sequence):
        residues_mask = np.zeros(len(sequence) - 2) * 1
        residues_mask[2] = 1
        zero = np.zeros(MAX_LENGTH - len(sequence) + 1)
        return np.concatenate((residues_mask, [0], zero), )

    def write_best_conformation(self, distance):
        if self.name not in self.best_conformations_dict.keys():
            self.best_conformations_dict[self.name] = [distance]
        else:
            self.best_conformations_dict[self.name].append(distance)

    def get_residue_distance(self, torsion, residue):
        residue_number = residue + 1
        if torsion == 0:
            current = self.protein_pose.phi(residue_number)
            target = self.target_protein_pose.phi(residue_number)

        else:
            current = self.protein_pose.psi(residue_number)
            target = self.target_protein_pose.psi(residue_number)
        angle_distance = self._get_residue_metric(np.radians(current), np.radians(target))
        return angle_distance / np.pi

    def _get_action_mask(self):
        offset = np.zeros(self.offset)
        mask = np.zeros(MAX_LENGTH - self.total_current_residue + 1 - self.offset)
        residue_mask = np.concatenate(
            ([0], offset, np.ones(self.total_current_residue - 2), mask))
        return np.array([residue_mask, np.array([1, 1]), np.ones(len(ANGLE_MOVE))])

    def _decode_action(self, action):
        angles_move_len = len(ANGLE_MOVE)
        angle = action % angles_move_len
        torsion = (action - angle) % (angles_move_len * 2) / angles_move_len
        amino = (action - angle - angles_move_len * torsion) / (angles_move_len * 2)
        return [amino, torsion, angle]

    def _norm_energy(self, energy, start_energy, final_energy):
        return (energy - final_energy) / (start_energy - final_energy)

    def get_distance_reward(self, current_distance):
        reward = (self.prev_distance - current_distance) / self.start_distance
        reward *= self.distance_reward_weight
        self.prev_distance = current_distance
        return reward

    def get_best_distance_reward(self, current_distance):
        reward = (self.best_distance - current_distance) / self.start_distance
        reward *= self.abs_distance_reward_weight
        self.best_distance = current_distance
        return reward

    def get_energy_reward(self, current_energy):
        reward = (self.prev_energy - current_energy) / self.start_energy
        reward *= self.energy_reward_weight
        self.prev_energy = current_energy
        return reward

    def get_torsion_reward(self, current_energy):
        reward = (self.prev_energy - current_energy) / self.start_energy
        reward *= self.energy_reward_weight
        self.prev_energy = current_energy
        return reward

    def step(self, action):
        self.done = False
        penalty = self._move(action)
        reward = penalty
        ob = self._get_state()

        energy = self.scorefxn(self.protein_pose)
        distance = self._get_distance(self.protein_pose, self.target_protein_pose)
        dist_reward = self.get_distance_reward(distance)
        energy_reward = self.get_energy_reward(energy)
        best_distance_rew = 0

        if self.best_distance > distance:
            best_distance_rew = self.get_distance_reward(distance)
        if energy < self.best_energy:
            self.best_energy = energy
            self.norm_best_energy = self._norm_energy(energy, self.start_energy, self.conform_energy)

        reward += (best_distance_rew + dist_reward + energy_reward)
        if distance < self.start_distance * 0.2:
            reward += 5
            self.done = True

        elif self.move_counter >= self.max_move_amount:
            try:
                reward += (self.start_distance - distance) / self.start_distance
            except:
                reward = 1
            self.done = True

        return [ob, reward, self.done,
                {"distance":distance,
                 "best": self.best_distance,
                 "name": self.name,
                 "start": self.start_distance,
                 'best_energy': self.norm_best_energy,
                 "final_distance": distance,
                 "current_energy": energy
                }]

    def set_default_pose(self, protein_pose):
        for i in range(1, protein_pose.total_residue() + 1):
            protein_pose.set_phi(i, 180)
            protein_pose.set_psi(i, 180)
        return protein_pose

    # def add_harder_protein(self):
    #     copyfile('protein_data/additional/3fpo.pdb', 'protein_data/short/3fpo.pdb')
    #     print("COPIED")

    def action_masks(self) -> List[bool]:
        actions = np.ones(MAX_LENGTH - 1 + 2 + len(ANGLE_MOVE)) * True
        actions[self.total_current_residue-self.input_shift:MAX_LENGTH-1] = False
        return actions

    def _init_metrics(self):
        self.start_distance = self._get_distance(self.protein_pose, self.target_protein_pose)
        self.prev_ca_rmsd = None
        self.achieved_goal_counter = 0
        self.best_distance = self._get_distance(self.protein_pose, self.target_protein_pose)
        self.prev_distance = self.start_distance

        self.best_energy = self.scorefxn(self.protein_pose)
        self.start_energy = self.scorefxn(self.protein_pose)
        self.prev_energy = self.scorefxn(self.protein_pose)
        self.conform_energy = self.scorefxn(self.target_protein_pose)
        self.norm_best_energy = 1

    def reset(self, protein_name=None):
        self.prev_residues_angle_distane = {}

        protein_directory = self.level_dir
        # if self.start_distance > self.best_distance:
        #     self.write_best_conformation(self.best_distance)
        if protein_name is None:
            protein_list = [protein for protein in os.listdir(protein_directory) if protein.endswith('.pdb')]
            self.name = np.random.choice(protein_list)
        else:
            self.name = protein_name
        protein_path = os.path.join(protein_directory, self.name)
        self.target_protein_pose = ProteinFoldEnv._library.get_protein(protein_path)
        self.protein_pose = self.set_default_pose(self.target_protein_pose.clone())
        for i in range(1, self.target_protein_pose.total_residue()):
            self.prev_residues_angle_distane[i * 2] = self.get_residue_distance(0, i)
            self.prev_residues_angle_distane[i * 2 + 1] = self.get_residue_distance(1, i)

        self.move_counter = 0
        self.reward = 0.0
        self.current_residue = 1

        self.encoded_residue_sequence = self._encode_residues(self.target_protein_pose.sequence())
        # self.save_best_matches()
        self.epoch_counter += 1
        # mask = self._get_action_mask()
        self._init_metrics()
        self.total_current_residue = self.protein_pose.total_residue()
        return self._get_state()

    def scramble_pose(self, pose):
        if np.random.rand() > 0.4:
            mask = np.random.rand(pose.total_residue())
            mask = mask > 0.4
            for x, residue in zip(mask, range(pose.total_residue() + 1)):
                if x:
                    noise = np.random.rand(2) * 25
                    pose.set_psi(residue + 1, self.target_protein_pose.psi(residue + 1) + noise[0])
                    pose.set_phi(residue + 1, self.target_protein_pose.phi(residue + 1) + noise[1])
        return pose

    def difference_energy(self):
        target_score = self.scorefxn(self.target_protein_pose)
        return (self.scorefxn(self.protein_pose) - target_score) / target_score

    def set_validation_mode(self, is_valid):
        self.validation = is_valid

    def seed(self, seed=None):
        np.random.seed(seed)

    def set_level(self, level_sub_dir):
        self.level_dir = os.path.join('protein_data', level_sub_dir)


if __name__ == '__main__':
    env = ProteinFoldEnv()
    env.reset()
    a = 5
    test = 'tt'