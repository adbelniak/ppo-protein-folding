import requests
import os

with open('protein_data/protein_list.txt') as f:
    for pdb in f:
        r = requests.get(pdb, allow_redirects=True)
        file_name = pdb.split('/')[-1]
        open(os.path.join('protein_data', 'short', file_name.lower()), 'wb').write(r.content)
