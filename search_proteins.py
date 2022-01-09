import requests
import os
import argparse
from pyrosetta import init, pose_from_pdb, pose_from_sequence
import pandas as pd
from pdb_cleaner import clean_protein_files

payload = {
  "query": {
    "type": "group",
    "logical_operator": "and",
    "nodes": [
      {
        "type": "group",
        "nodes": [
          {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "attribute": "rcsb_entry_info.selected_polymer_entity_types",
              "operator": "exact_match",
              "negation": False,
              "value": "Protein (only)"
            }
          },
          {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "attribute": "entity_poly.rcsb_sample_sequence_length",
              "operator": "less",
              "negation": False,
              "value": 17
            }
          },
          {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "attribute": "rcsb_entry_info.polymer_entity_count_protein",
              "operator": "equals",
              "negation": False,
              "value": 1
            }
          },
          {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
              "operator": "equals",
              "negation": False,
              "value": 1
            }
          }
        ],
        "logical_operator": "and"
      },
      {
        "type": "terminal",
        "service": "text",
        "parameters": {
          "attribute": "entity_poly.rcsb_entity_polymer_type",
          "operator": "exact_match",
          "value": "Protein"
        }
      },
      {
        "type": "terminal",
        "service": "text",
        "parameters": {
          "attribute": "rcsb_entity_source_organism.ncbi_parent_scientific_name",
          "operator": "exact_match",
          "value": "Eukaryota"
        }
      },
      {
        "type": "terminal",
        "service": "text",
        "parameters": {
          "attribute": "rcsb_entity_source_organism.ncbi_scientific_name",
          "operator": "exact_match",
          "value": "Homo sapiens"
        }
      }
    ],
    "label": "text"
  },
  "return_type": "entry",
  "request_options": {
    "pager": {
      "start": 0,
      "rows": 1000
    },
    "scoring_strategy": "combined",
    "sort": [
      {
        "sort_by": "score",
        "direction": "desc"
      }
    ]
  }
}

def search_proteins():
    base_url = 'https://search.rcsb.org/rcsbsearch/v1/query'
    r = requests.post(base_url, allow_redirects=True, json=payload)
    protein_list = []
    if r.status_code == 200:
        data = r.json()
        for protein in data['result_set']:
            protein_list.append(protein['identifier'])
    else:
        print(r.content)
    return protein_list


def download_proteins(saving_directory, protein_name_list):
    base_url = 'https://files.rcsb.org/download/'
    for pdb in protein_name_list:
        protein_name = pdb + '.pdb'
        url = base_url + protein_name
        r = requests.get(url, allow_redirects=True)
        print("downloading pdb file:" + protein_name)
        open(os.path.join(saving_directory, protein_name.lower()), 'wb').write(r.content)


def create_dataset_description(dataset_path, descr_file_path):
    init()
    protein_length_counter = {}
    for protein_path in os.listdir(dataset_path):
        if protein_path.endswith('.pdb'):
            protein = pose_from_pdb(os.path.join(dataset_path, protein_path))
            protein_length_counter[protein_path] = [protein.total_residue(), protein.sequence()]
    protein_list_df = pd.DataFrame.from_dict(protein_length_counter, orient='index', columns=['residue_number', 'sequence'])
    protein_list_df.index.rename('protein_name')
    print(protein_list_df)
    path_to_df = os.path.join(dataset_path, descr_file_path)
    protein_list_df.to_csv(path_to_df, index_label='protein_name')
    return protein_list_df


def split_dataset(dataset_path, descr_file_path):
    init()
    path_to_df = os.path.join(dataset_path, descr_file_path)

    if not os.path.isfile(path_to_df):
        protein_list_df = create_dataset_description(dataset_path, descr_file_path)
    else:
        protein_list_df = pd.read_csv(path_to_df, index_col='protein_name')

    train = protein_list_df.groupby('residue_number').sample(frac=0.85, random_state=0)
    train.reset_index(inplace=True)
    test = protein_list_df[~protein_list_df.index.isin(train['protein_name'])]
    train.to_csv(os.path.join(dataset_path, 'train.csv'))
    test.to_csv(os.path.join(dataset_path, 'test.csv'))

def create_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def arg_parse():
    parser = argparse.ArgumentParser(description='Protein downloader.')
    parser.add_argument('--download_directory', dest='download_directory', action='store', default='protein_data/baseline_temp', type=str)
    parser.add_argument('--dest_directory', dest='dest_directory', action='store', default='protein_data/baseline', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    descr_file_path = 'protein_info.csv'
    create_if_not_exists(args.download_directory)
    create_if_not_exists(args.dest_directory)

    protein_name_list = search_proteins()
    print(len(protein_name_list))

    download_proteins(args.download_directory, protein_name_list)
    clean_protein_files(args.download_directory, args.dest_directory, 17)
    create_dataset_description(args.dest_directory, descr_file_path)
    split_dataset(args.dest_directory, descr_file_path)