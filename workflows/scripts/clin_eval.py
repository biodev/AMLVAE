import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sbn
import numpy as np 

import torch

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score
import umap
from sklearn.decomposition import PCA
from amlvae.models.VAE import VAE 

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

                            
    return argparser.parse_args()

def load(root): 

    data = pd.read_csv(f'{root}/aml_expr.csv')
    data = data.set_index(data.columns[0])
    partitions = torch.load(f'{root}/aml_partitions.pt', weights_only=False)
    X_train = torch.tensor( 
        data.loc[partitions['train_ids'], :].values, dtype=torch.float32
    )
    X_val = torch.tensor(
        data.loc[partitions['val_ids'], :].values, dtype=torch.float32
    )
    X_test = torch.tensor(
        data.loc[partitions['test_ids'], :].values, dtype=torch.float32
    )

    return X_train, X_val, X_test


if __name__ == '__main__': 

    print()
    print('---------------------------------------------')
    print('VAE: Evaluation and embedding visualization')
    print('---------------------------------------------')
    print() 
    print('arguments:')
    args = get_args()
    print(args)
    print('---------------------------------------------')
    
    X_train, X_val, X_test = load(args.proc) 
    model = torch.load(args.model_path, weights_only=False, map_location='cuda')
    model = model.eval()


    # load cluster data (source: ??)
    tcga_df = pd.read_excel(f'{args.data}/TCGA.LAML.cNMF-clustering.20140820.xlsx', sheet_name='mRNA-seq (n=179)')
    gdc2tcga = pd.read_csv(f'{args.data}/tcga_sample_mapping.csv').rename({'patient_barcode':'sample.id'}, axis=1)
    tcga_df = tcga_df.merge(gdc2tcga, on='sample.id', how='left')

    expr = pd.read_csv(f'{args.proc}/aml_expr.csv')
    X = expr.iloc[:, 1:].values
    z = model.encode(torch.tensor(X, dtype=torch.float32).cuda())[0].cpu().detach().numpy()

    reducer = umap.UMAP(metric='cosine', n_neighbors=15, min_dist=0.1, spread=1.0, random_state=42) 
    u = reducer.fit_transform(z)

    res = pd.DataFrame(u, columns=['u1', 'u2']).assign(gdc_id=expr.iloc[:, 0].values)
    res = res.merge(tcga_df, on='gdc_id', how='left')
    res = res.fillna('NA')

    clin = pd.read_csv(f'{args.data}/beataml_clinical_for_inputs.csv')
    res = res.merge(clin, on='gdc_id', how='left')

    f, axes = plt.subplots(1,1, figsize=(12, 12))
    sbn.scatterplot(data=res[lambda x: x.cluster != 'NA'], x='u1', y='u2', hue='cluster', alpha=1., palette='tab10', ax=axes)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f'{args.out}/embedding_umap_plot__tcga_cluster.png', dpi=300, bbox_inches='tight')

    f, axes = plt.subplots(1,1, figsize=(12, 12))
    sbn.scatterplot(data=res, x='u1', y='u2', hue='fabBlastMorphology', alpha=1., palette='tab10', ax=axes)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f'{args.out}/embedding_umap_plot__fabBlastMorphology.png', dpi=300, bbox_inches='tight')

    f, axes = plt.subplots(1,1, figsize=(12, 12))
    sbn.scatterplot(data=res, x='u1', y='u2', hue='consensusAMLFusions', alpha=1., palette='tab10', ax=axes)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f'{args.out}/embedding_umap_plot__consensusAMLFusions.png', dpi=300, bbox_inches='tight')

    f, axes = plt.subplots(1,1, figsize=(12, 12))

    sbn.scatterplot(data=res[lambda x: ~x.overallSurvival.isna()], x='u1', y='u2', hue='overallSurvival', alpha=1., ax=axes)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f'{args.out}/embedding_umap_plot__overallSurvival.png', dpi=300, bbox_inches='tight')

    f, axes = plt.subplots(model.latent_dim, model.latent_dim, figsize=(3*model.latent_dim, 3*model.latent_dim))
    for i in range(model.latent_dim): 
        for j in range(model.latent_dim): 
            if i == j: 
                sbn.histplot(data=res, x=z[:,i], ax=axes[i,j], bins=25)
            elif i < j: 
                sbn.scatterplot(data=res, x=z[:,i], y=z[:,j], ax=axes[i,j], hue='consensusAMLFusions', alpha=0.5)
    plt.savefig(f'{args.out}/embedding_latent_space.png', dpi=300, bbox_inches='tight')
            

    print('evaluation complete.')
    print('---------------------------------------------')
    print()