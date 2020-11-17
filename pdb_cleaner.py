from pdbtools.clean import pdbClean
import os
from pyrosetta import init, pose_from_pdb, pose_from_sequence
from pyrosetta.rosetta.core.scoring import CA_rmsd
import pdbtools
from shutil import copyfile

init()
def clean_protein_files(dir: str, out_dir: str):
    for file in os.listdir(dir):
        try:
            f = open(os.path.join(dir,file), 'r')
            print(file)
            pdb = f.readlines()
            f.close()

            pdb = pdbClean(pdb, renumber_residues=True)

            g = open(os.path.join(dir, file), "w")
            g.writelines(pdb)
            g.close()
            target_protein_pose = pose_from_pdb(os.path.join(dir, file))

            if target_protein_pose.total_residue() < 2:
                os.remove(os.path.join(dir, file))
        except Exception as e:
            print(e)
            pass

def sort_pdbs(dir: str):
    for x in range(2, 7):
        if not os.path.exists('protein_data/short_' + str(x)):
            os.makedirs('protein_data/short_' + str(x))
    for file in os.listdir(dir):
        target_protein_pose = pose_from_pdb(os.path.join(dir, file))
        dir_to_move = 'short_' + str(target_protein_pose.total_residue())
        copyfile(os.path.join(dir, file), os.path.join('protein_data', dir_to_move, file))


if __name__ == '__main__':
    # clean_protein_files('protein_data/6_length', 'protein_data/short_valid')
    sort_pdbs('protein_data/6_length')