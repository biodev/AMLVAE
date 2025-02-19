
import numpy as np 


class ExprProcessor(): 
    def __init__(self, expr_long, target='fpkm_uq_unstranded', norm='zscore_log2', top_genes=1000, idxs=None):
        
        self.target = target 
        expr = expr_long[['id', 'gene_name', target]].groupby(['id', 'gene_name']).mean()
        expr = expr.reset_index().pivot(index='id', columns='gene_name', values=target)

        if norm == 'zscore_log2':
            expr = np.log2(expr + 1)
            mu = expr.mean(axis=0)
            sd = expr.std(axis=0)
            self.transform_params = {'mu': mu, 'sd': sd, 'norm': norm}

            expr = (expr - mu) / (sd + 1e-8)

        else: 
            raise ValueError('norm not implemented')
        

        if idxs is None:
            self.idxs = np.argsort(expr.var(axis=0))[::-1][:top_genes]
        else: 
            self.idxs = idxs 

        self.expr = expr.iloc[:, self.idxs]

        self.gene_names = self.expr.columns
        self.ids = self.expr.index.values

    def get_data(self):
        return self.expr.values, self.ids
    
    def get_gene_names(self):
        return self.gene_names 
    
    def process_new(self, expr_long): 
        '''process a new dataset'''

        expr = expr_long[['id', 'gene_name', self.target]].groupby(['id', 'gene_name']).mean()
        expr = expr.reset_index().pivot(index='id', columns='gene_name', values=self.target)

        if self.transform_params['norm'] == 'zscore_log2':
            expr = np.log2(expr + 1)
            expr = (expr - self.transform_params['mu']) / (self.transform_params['sd'] + 1e-8)

        return expr.iloc[:, self.idxs].values, expr.index.values