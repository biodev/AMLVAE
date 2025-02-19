import numpy as np 

class MutProcessor(): 
    def __init__(self, mut, top_genes=1000, idxs=None):
        
        if idxs is None:
            self.idxs = np.argsort(mut.var(axis=0))[::-1][:top_genes]
        else: 
            self.idxs = idxs 

        self.mut = mut.iloc[:, self.idxs]

        self.gene_names = self.mut.columns
        self.ids = self.mut.index

    def get_data(self):
        return self.mut.values, self.ids.values
    
    def get_gene_names(self):
        return self.gene_names 
    
    def process_new(self, mut_long): 
        '''process a new dataset'''

        pass 