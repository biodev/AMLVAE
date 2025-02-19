import numpy as np 
from sklearn.preprocessing import LabelEncoder

class ClinProcessor: 
    def __init__(self, clin, targets):

        self.clin = clin 
        self.targets = targets 

        targ2cat = {}
        for targ in self.targets: 
            x = self.clin[targ].values
            le = LabelEncoder()
            le.fit(x)
            targ2cat[targ] = le
        self.targ2cat = targ2cat 

    def get_data(self, ids=None, id_type=None): 

        assert ((ids is None) and (id_type is None)) or ((ids is not None) and (id_type is not None)), 'ids and id_type should either be both passed or both be None'
        
        outs = [] 
        for targ in self.targets:
            if (ids is not None) & (id_type is not None): 
                clin = self.clin.set_index(id_type)
                clin = clin.loc[ids]
            else: 
                clin = self.clin 

            l = clin[targ].values
            l = self.targ2cat[targ].transform(l)
            n = len(self.targ2cat[targ].classes_)
            x = np.zeros((len(l), n))
            x[np.arange(len(l)), l] = 1
            outs.append(x)
        
        return np.concatenate(outs, axis=1)