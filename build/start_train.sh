#/bin/bash

docker run --name protein_training -d -it -v "${HOME}"/ppo-protein-folding:"$(pwd)" -v "$(pwd)"/protein_data:"$(pwd)"/protein_data -v "$(pwd)"/log:"$(pwd)"/log -p 5001:6006 ppo-protein

docker exec -i protein_training /bin/bash -c "python3  download_proteins.py && python3 pdb_cleaner.py"

docker exec -it -d protein_training tensorboard --logdir=. --host 0.0.0.0