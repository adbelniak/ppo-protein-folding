import json
import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from pyrosetta import init, pose_from_pdb, pose_from_sequence
from pyrosetta.rosetta.core.scoring import CA_rmsd
import numpy as np
from angles import normalize
from pyrosetta.teaching import *

ANGLE_MOVE = [-90, -45, -10, -5, 5, 10, 45, 90]
import logging

logger = logging.getLogger(__name__)

RESIDUE_LETTERS = [
    'R', 'H', 'K',
    'D', 'E',
    'S', 'T', 'N', 'Q',
    'C', 'G', 'P',
    'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'empty'
]

MAX_LENGTH = 32


class ProteinFoldEnv(gym.Env, utils.EzPickle):

    def __init__(self, max_move_amount=1000):
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
            # "energy": spaces.Box(low=np.array([-np.inf]), high=np.array([np.inf]), dtype=np.float32),
            "backbone": spaces.Box(low=-3, high=3, shape=(MAX_LENGTH, 3,)),
            "step_to_end": spaces.Discrete(1),
            "amino_acid": spaces.Box(low=-1, high=1, shape=(MAX_LENGTH, len(RESIDUE_LETTERS),)),
            "current_distance": spaces.Box(low=-10, high=1, shape=(1,))
        })
        self.action_space = spaces.MultiDiscrete([MAX_LENGTH, 2, len(ANGLE_MOVE)])
        self.epoch_counter = 0
        # self.action_space = spaces.Dict({
        #     "angles": spaces.Box(low=-1, high=1, shape=(3,)),
        #     "torsion": spaces.Discrete(3),
        #     "residue": spaces.Discrete(MAX_LENGTH)
        # })
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
        self.level_dir = 'protein_data/short_3'
        self.offset = 0

    def _configure_environment(self):
        """
        Provides a chance for subclasses to override this method and supply
        a different server configuration. By default, we initialize one
        offense agent against no defenders.
        """
        self._start_rossetta()

    def _start_rossetta(self):
        init(extra_options="-mute all")

    def _get_ca_metric(self, coordinates, target):
        return CA_rmsd(coordinates, target)

    def _move_on_torsion(self, residue_number, move_pose, get_angle, set_angle):
        new_torsion_position = normalize(get_angle(residue_number) + move_pose, -180, 180)
        set_angle(residue_number, new_torsion_position)

    def _move(self, action):
        if action is not None:
            self.move_counter += 1
            # residue_number = self.current_residue + 1
            residue_number = action[0] + 1
            torsion_number = action[1]
            move_pose_index = action[2]
            move_pose = ANGLE_MOVE[move_pose_index]

            if residue_number < self.protein_pose.total_residue() + 1:
                if torsion_number == 0:
                    self._move_on_torsion(residue_number, move_pose, self.protein_pose.phi, self.protein_pose.set_phi)
                if torsion_number == 1:
                    self._move_on_torsion(residue_number, move_pose, self.protein_pose.psi, self.protein_pose.set_psi)
                if torsion_number == 2:
                    self.current_residue += 1
                    if self.current_residue >= self.protein_pose.total_residue() - 1:
                        self.current_residue = 1
                    return -0.0
                return 0.0
            else:
                return -0.1
        return 0.0

    def _encode_residues(self, sequence):
        encoded_residues = []
        for res in sequence:
            temp = np.zeros(len(RESIDUE_LETTERS))
            temp[RESIDUE_LETTERS.index(res)] = 1
            encoded_residues.append(temp)
        for zero in range(MAX_LENGTH - len(sequence)):
            temp = np.zeros(len(RESIDUE_LETTERS))
            temp[len(RESIDUE_LETTERS) - 1] = -1

            encoded_residues.append(temp)
        return encoded_residues

    def _get_residue_metric(self, resiude_pose, target_residue_pose):
        angle_distance = np.arctan2(np.sin(resiude_pose - target_residue_pose),
                                    np.cos(resiude_pose - target_residue_pose))
        return abs(angle_distance)

    def _get_state(self):
        # psis = np.divide(
        #     self.target_protein_pose.psi(self.current_residue + 1) - self.protein_pose.psi(self.current_residue + 1),
        #     180.0)
        psis = np.divide([self.target_protein_pose.psi(i + 1) - self.protein_pose.psi(i + 1) for i in
                          range(self.protein_pose.total_residue())], 180.0)
        phis = np.divide([self.target_protein_pose.phi(i + 1) - self.protein_pose.phi(i + 1) for i in
                          range(self.protein_pose.total_residue())], 180.0)

        # omegas = np.divide([self.target_protein_pose.omega(i + 1) - self.protein_pose.omega(i + 1) for i in range(self.protein_pose.total_residue())], 180.0)
        # phis = np.divide(
        #     self.target_protein_pose.phi(self.current_residue + 1) - self.protein_pose.phi(self.current_residue + 1),
        #     180.0)
        rest_zeros = MAX_LENGTH - len(self.target_protein_pose.sequence())
        psis = np.concatenate((psis, np.zeros(rest_zeros)))
        phis = np.concatenate((phis, np.zeros(rest_zeros)))
        one_hot = np.zeros(MAX_LENGTH)
        one_hot[self.current_residue + 1] = 1
        backbone_geometry = [[psi, phi, one] for psi, phi, one in
                             zip(psis, phis, one_hot)]
        distance = self._get_ca_metric(self.protein_pose, self.target_protein_pose)

        return {
            "backbone": backbone_geometry,
            # "energy": [self.difference_energy()],
            "step_to_end": (32 - self.move_counter) / 32,
            "amino_acid": self.encoded_residue_sequence,
            "current_distance": [distance]
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
        mask = np.zeros(MAX_LENGTH - self.protein_pose.total_residue() - 1)
        residue_mask = np.concatenate((np.ones(self.protein_pose.total_residue() + 1), mask))
        return np.array([residue_mask, np.array([1, 1]), np.ones(len(ANGLE_MOVE))])

    def step(self, action):
        self.done = False
        penalty = self._move(action)
        # done = False
        reward = penalty
        torsion = action[1]
        if torsion < 2:
            current_angle_distance = self.get_residue_distance(torsion, self.current_residue)

            # reward += self.prev_residues_angle_distane[
            #               (self.current_residue + 1) * 2 + torsion] - current_angle_distance
            self.prev_residues_angle_distane[(self.current_residue + 1) * 2 + torsion] = current_angle_distance
        # if not self.move_counter % 4:
        #     self.current_residue += 1
        # if self.current_residue >= self.protein_pose.total_residue() - 1:
        #     self.current_residue = 1
        ob = self._get_state()

        # reward = 0
        energy = self.scorefxn(self.protein_pose)
        distance = self._get_ca_metric(self.protein_pose, self.target_protein_pose)
        # if self.prev_energy:
        #     reward += (self.prev_energy - energy) / self.start_energy
        if self.prev_ca_rmsd:
            reward += 0.1 * (self.prev_ca_rmsd - distance) / self.start_distance

        if self.best_distance > distance:
            # if distance < self.start_distance * 0.5:
            #     reward += 0.5
            reward += (self.best_distance - distance) / self.start_distance
            self.best_distance = distance
            self.best_energy = energy

        self.prev_ca_rmsd = distance
        self.prev_energy = energy

        if distance < self.start_distance * 0.2:
            # reward +=  self.start_distance - distance / self.start_distance
            reward += 4
            self.done = True

        if self.move_counter >= 64:
            # reward -= distance / self.start_distance
            self.done = True
        mask = self._get_action_mask()
        return [ob, reward, self.done,
                {"best": self.best_distance, "name": self.name, "start": self.start_distance, 'action_mask': mask}]

    def set_default_pose(self, protein_pose):
        for i in range(1, protein_pose.total_residue()):
            protein_pose.set_phi(i, 180)
            protein_pose.set_psi(i, 180)
        return protein_pose

    # def add_harder_protein(self):
    #     copyfile('protein_data/additional/3fpo.pdb', 'protein_data/short/3fpo.pdb')
    #     print("COPIED")

    def reset(self):
        try:
            dir = self.level_dir
            self.offset = np.random.randint(0, 10)
            protein_name = np.random.choice(os.listdir(dir))
            if self.start_distance > self.best_distance:
                self.write_best_conformation(self.best_distance)
            self.name = protein_name

            self.target_protein_pose = pose_from_pdb(os.path.join(dir, self.name))
            self.prev_residues_angle_distane = {}
            self.protein_pose = self.set_default_pose(pose_from_pdb(os.path.join(dir, self.name)))
            for i in range(1, self.target_protein_pose.total_residue()):
                self.prev_residues_angle_distane[i * 2] = self.get_residue_distance(0, i)
                self.prev_residues_angle_distane[i * 2 + 1] = self.get_residue_distance(1, i)

            # if not self.shuffle:
            #     self.protein_pose = pose_from_sequence(self.target_protein_pose.sequence())
            # else:
            self.start_distance = self._get_ca_metric(self.protein_pose, self.target_protein_pose)

            # self.scramble_pose(self.protein_pose)
            self.move_counter = 0
            self.reward = 0.0
            self.prev_ca_rmsd = None
            self.achieved_goal_counter = 0
            self.current_residue = 1

            self.best_distance = self._get_ca_metric(self.protein_pose, self.target_protein_pose)

            self.best_energy = self.scorefxn(self.protein_pose)
            self.start_energy = self.scorefxn(self.protein_pose)
            self.prev_energy = self.scorefxn(self.protein_pose)

            self.encoded_residue_sequence = self._encode_residues(self.target_protein_pose.sequence())
            self.save_best_matches()
            self.epoch_counter += 1
            mask = self._get_action_mask()
            return self._get_state(), mask
        except:
            print(self.name)

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
