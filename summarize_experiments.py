import os
import numpy as np
import re

import pandas as pd


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def best_model_from_path(experiment_dir, model_prefix):
    models_list = []
    for file in os.listdir(experiment_dir):
        if file.endswith('.zip') and file.startswith(model_prefix):
            models_list.append(file)

    if len(models_list) == 0:
        return None

    models_list.sort(key=natural_keys)
    return models_list[-1]


if __name__ == '__main__':
    experiments_dir = 'logs'
    model_prefixes = ['best_model', 'best_distance_model']
    models_list = []

    for exp_folder in os.listdir(experiments_dir):
        single_exp_path = os.path.join(experiments_dir, exp_folder)
        if os.path.isdir(single_exp_path):
            for pref in model_prefixes:
                best_model = best_model_from_path(single_exp_path, pref)
                if best_model:
                    best_model_path = os.path.join(single_exp_path, best_model)
                    models_list.append([best_model_path, single_exp_path.split('/')[-1]])
                    # models_list.append(single_exp_path.split('/')[-1])

    pd.DataFrame(models_list).to_csv("models_to_download.csv")
