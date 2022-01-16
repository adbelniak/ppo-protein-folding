import requests
import os

with open('protein_data/protein_list.txt') as f:
    base_url = 'https://files.rcsb.org/download/'
    for pdb in f.read().split(','):
        protein_name = pdb + '.pdb'
        url = base_url + protein_name
        r = requests.get(url, allow_redirects=True)
        print("downloading pdb file:" + protein_name)
        open(os.path.join('protein_data', '6_length', protein_name.lower()), 'wb').write(r.content)
