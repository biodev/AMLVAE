import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
import argparse 
import umap
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np

def get_args(): 

    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('--z_path', type=str, default='/path/to/dataset_z.csv', 
                           help='path to embedding file')
    argparser.add_argument('--clin_path', type=str, default='/path/to/clin.xlsx', 
                           help='path to embedding file')
    argparser.add_argument('--out', type=str, default='../output/',
                            help='path to output dir')
    argparser.add_argument('--n_neighbors', type=int, default=15,
                            help='number of neighbors for UMAP')
    argparser.add_argument('--min_dist', type=float, default=0.1,
                            help='minimum distance for UMAP')
    argparser.add_argument('--metric', type=str, default='euclidean',
                            help='metric for UMAP')
    argparser.add_argument('--clin_vars', type=str, default='',
                            help='clinical variables to plot')
    argparser.add_argument('--seed', type=int, default=42,
                            help='random seed for reproducibility')
    args = argparser.parse_args()

    args.clin_vars = args.clin_vars.split('<::>')
    return args 


if __name__ == '__main__': 

    print()
    print('---------------------------------------------')
    print('VAE: embedding visualization by clinical features')
    print('---------------------------------------------')
    print() 
    print('arguments:')
    args = get_args()
    print(args)
    print('---------------------------------------------')

    # seed 
    np.random.seed(args.seed)

    print('loading data...')
    
    z = pd.read_csv(args.z_path, index_col=0) 
    
    if args.clin_path.endswith('.xlsx'):
        clin = pd.read_excel(args.clin_path, sheet_name=0)
        id_name = 'array_id'
    elif args.clin_path.endswith('.csv'): 
        clin = pd.read_csv(args.clin_path)
        id_name = 'gdc_id'
    else:
        raise ValueError('Unsupported clinical data format. Use .xlsx or .csv.')
    
    print() 
    print(clin.head())
    print() 

    print('running umap...')
    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, n_components=2, metric=args.metric)
    u = reducer.fit_transform(z.values) 
    u = pd.DataFrame(u, index=z.index, columns=['u1','u2']).assign(**{id_name: z.index}) 
    u = u.merge(clin, on=id_name, how='left') 

    for clin_var in args.clin_vars: 
        print('plotting UMAP for', clin_var)

        try: 
            plt.figure(figsize=(8, 8)) 
            sbn.scatterplot(data=u, x='u1', y='u2', hue=clin_var)
            plt.title(f'UMAP; {clin_var}')
            # place legend to the right of the plot
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.savefig(f'{args.out}/umap_{clin_var}.png', dpi=300, bbox_inches='tight')
            plt.close()
        except: 
            print(f'\tError plotting {clin_var}. Skipping...')

    # mark complete. 
    with open(f'{args.out}/clin_viz_complete.txt', 'w') as f: 
        f.write('complete')

    print('viz complete')
    print('---------------------------------------------')
    print()
    



    


    