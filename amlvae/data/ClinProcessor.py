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

    def get_data(self, ids=None): 

        outs = {}
        for targ in self.targets:
            clin = self.clin.loc[ids]

            l = clin[targ].values
            l = self.targ2cat[targ].transform(l)
            n = len(self.targ2cat[targ].classes_)
            x = np.zeros((len(l), n))
            x[np.arange(len(l)), l] = 1
            outs[targ] = x
        
        return outs