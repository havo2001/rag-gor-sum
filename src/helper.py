import datetime
import random
import numpy as np
import torch
import os

import networkx as nx
import pickle


def show_time():
    time_stamp = '\033[1;31;40m[' + str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ']\033[0m'

    return time_stamp


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def store_nx(nx_obj, path):
    nx.write_graphml_lxml(nx_obj, path)


def write_to_pkl(data, output_file):
    with open(output_file, 'wb') as file:
        pickle.dump(data, file)


def load_nx(path) -> nx.Graph:
    return nx.read_graphml(path)


def read_from_pkl(output_file):
    with open(output_file, 'rb') as file:
        data = pickle.load(file)

    return data
