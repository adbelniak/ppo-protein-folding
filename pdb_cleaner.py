
from pdbtools.clean import pdbClean
import os
from pyrosetta import init, pose_from_pdb, pose_from_sequence
from pyrosetta.rosetta.core.scoring import CA_rmsd
import pdbtools
from shutil import copyfile
from pyrosetta.toolbox import cleanATOM
init()
def clean_protein_files(dir: str, out_dir: str, max_res):
    for file in os.listdir(dir):
        try:
            f = open(os.path.join(dir,file), 'r')
            print(file)
            pdb = f.readlines()
            f.close()

            pdb = pdbClean(pdb, renumber_residues=True)

            g = open(os.path.join(out_dir, file), "w")
            g.writelines(pdb)
            g.close()
            target_protein_pose = pose_from_pdb(os.path.join(out_dir, file))

            if target_protein_pose.total_residue() < 2 or target_protein_pose.total_residue() > max_res:
                os.remove(os.path.join(out_dir, file))
        except Exception as e:
            print(e)
            print("REMOVED demaged protein ", file)
            os.remove(os.path.join(dir, file))
            pass


def sort_pdbs(src:dir, dir: str):
    for x in range(3, 17):
        if not os.path.exists(f'{dir}/short_' + str(x)):
            os.makedirs(f'{dir}/short_' + str(x))
    for file in os.listdir(src):
        if file.endswith('.pdb'):
            target_protein_pose = pose_from_pdb(os.path.join(src, file))
            dir_to_move = 'short_' + str(target_protein_pose.total_residue())
            copyfile(os.path.join(src, file), os.path.join(dir, dir_to_move, file))


if __name__ == '__main__':
    # clean_protein_files('protein_data/6_length', 'protein_data/short_valid')
    sort_pdbs('protein_data/baseline', 'protein_data/benchmark')