import numpy as np
from pyrosetta import init, pose_from_pdb, pose_from_sequence
from pyrosetta.rosetta.core.scoring import CA_rmsd
import os

init()
amino = ['A', 'C']

def create_simple_benchmark_data():
    start_sequence = 'AC'

    for x in range(1, 10):
        dir_path = os.path.join('benchmark', 'bench_' + str(x))
        os.mkdir(dir_path)
        sequence = start_sequence
        for i in range (x + 1):
            sequence = sequence + amino[i % 2]

        protein = pose_from_sequence(sequence)
        protein.pdb_info().name("protein_" + str(x))

        for i in range(0, x+1):
            # protein.set_phi(i + 1, 25 * (1 + i % x))
            # protein.set_psi(i + 1, 25 * (1 + i % x))
            protein.set_phi(i + 1, 25 )
            protein.set_psi(i + 1, 25 )
            print(protein.phi(i + 1))
        protein.dump_pdb(os.path.join(dir_path, "protein_" + str(x) + '.pdb'))

def create_incremental_benchmark_data():
    start_sequence = 'A'

    for x in range(5, 6):
        dir_path = 'benchmark_incremental'
        sequence = start_sequence
        for i in range(x + 1):
            sequence = sequence + 'A'
        sequence = sequence + 'A'
        protein = pose_from_sequence(sequence)
        protein.pdb_info().name("protein_" + str(x))

        for i in range(0, x):
            # protein.set_phi(i + 1, 25 * (1 + i % x))
            # protein.set_psi(i + 1, 25 * (1 + i % x))
            protein.set_phi(i + 1, 135)
            protein.set_psi(i + 1, 135)

        protein.set_phi(protein.total_residue() - 1, 100)
        protein.set_psi(protein.total_residue() - 1, 100)
        protein.dump_pdb(os.path.join(dir_path, "protein_" + str(x) + '.pdb'))


if __name__ == '__main__':
    create_incremental_benchmark_data()