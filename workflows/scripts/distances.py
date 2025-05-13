import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
import argparse 


def get_args(): 

    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('--data', type=str, default='../../data/', 
                           help='path to data dir')
    argparser.add_argument('--proc', type=str, default='../proc/',
                            help='path to proc dir')
    argparser.add_argument('--out', type=str, default='../output/',
                            help='path to output dir')
    argparser.add_argument('--model_path', type=str, default=None,
                            help='path to model dir')
    argparser.add_argument('--dataset', type=str, default='aml',
                            help='dataset name')
    argparser.add_argument('--metric', type=str, default='euclidean',
                            help='metric for pairwise distance calculation')

    return argparser.parse_args()

def load(args): 

    data = pd.read_csv(f'{args.proc}/{args.dataset}_expr.csv')
    data = data.set_index(data.columns[0])
    X = torch.tensor( data.values, dtype=torch.float32)
    return X, data.index


if __name__ == '__main__': 

    print()
    print('---------------------------------------------')
    print('VAE: Computing pairwise distances')
    print('---------------------------------------------')
    print() 
    print('arguments:')
    args = get_args()
    print(args)
    print('---------------------------------------------')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X, ids = load(args) 
    model = torch.load(args.model_path, weights_only=False, map_location=device)
    model = model.eval()

    with torch.no_grad():
        z = model.encode(X.to(device))[0].cpu().numpy() 

    z_df = pd.DataFrame(z, index=ids, columns=[f'z{i+1}' for i in range(z.shape[1])])
    z_df.to_csv(f'{args.out}/{args.dataset}_z.csv', index=True, header=True)

    # calculate pairwise distances 
    print('Calculating pairwise distances')
    dist = pairwise_distances(z, metric=args.metric) 
    dist = pd.DataFrame(dist, index=ids, columns=ids) 

    dist.to_csv(f'{args.out}/{args.dataset}_dist.csv', index=True, header=True) 



    


    