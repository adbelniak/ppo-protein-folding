from pdbtools.clean import pdbClean
import os
import pdbtools
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
        except Exception as e:
            print(e)
            pass

if __name__ == '__main__':
    clean_protein_files('protein_data/short', 'protein_data/short_valid')