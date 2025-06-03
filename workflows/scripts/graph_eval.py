import pandas as pd 
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse import coo_matrix
import argparse 
import networkx as nx 
from matplotlib import pyplot as plt
import pickle as pkl

def get_args(): 

    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('--run_dir', type=str, default='../../runs/',
                            help='path to run dir')
    argparser.add_argument('--out', type=str, default='../output/',
                            help='path to output dir')
    argparser.add_argument('--dataset', type=str, default='aml',
                            help='dataset name')
    argparser.add_argument('--num_folds', type=int, default=5,
                            help='number of folds')
    argparser.add_argument('--k', type=int, default=20,
                            help='number of nearest neighbours')
    argparser.add_argument('--mu', type=float, default=0.5,
                            help='scaling factor for adaptive Gaussian kernel bandwidth')
    argparser.add_argument('--T', type=int, default=20,
                            help='number of cross-network diffusion iterations')
    argparser.add_argument('--edge_thr_q', type=float, default=0.95,
                            help='post-fusion pruning threshold quantile for edge weights')
    argparser.add_argument('--seed', type=int, default=42,
                            help='random seed for reproducibility')

    return argparser.parse_args()