import requests
import os
import argparse

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


def arg_parse():
    parser = argparse.ArgumentParser(description='Protein downloader.')
    parser.add_argument('--saving_directory', dest='saving_directory', action='store', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    protein_name_list = search_proteins()
    print(len(protein_name_list))
    download_proteins(args.saving_directory, protein_name_list)
    clean_protein_files('protein_data/baseline', 'protein_data/baseline', 17)
