import torch

class MLP(torch.nn.Module): 

    def __init__(self, in_channels, hidden_channels, out_channels, layers=2, dropout=0, 
                        nonlin=torch.nn.ELU, out=None, norm=torch.nn.LayerNorm, bias=True): 
        '''
        
        Args: 
            in_channels             int                 number of input channels 
            hidden_channels         int                 number of hidden channels per layer 
            out_channels            int                 number of output channels 
            layers                  int                 number of hidden layers 
            dropout                 float               dropout regularization probability 
            nonlin                  pytorch.module      non-linear activation function 
            out                     pytorch.module      output transformation to be applied 
            norm                    pytorch.module      normalization method to use 
        '''
        super().__init__()
        
        seq = [torch.nn.Linear(in_channels, hidden_channels, bias=bias)]
        if norm is not None: seq.append(norm(hidden_channels))
        seq += [nonlin(), torch.nn.Dropout(dropout)] 
        for _ in range(layers - 1): 
            seq += [torch.nn.Linear(hidden_channels, hidden_channels, bias=bias)]
            if norm is not None: seq.append(norm(hidden_channels))
            seq += [nonlin(), torch.nn.Dropout(dropout)]
        seq += [torch.nn.Linear(hidden_channels, out_channels, bias=bias)]
        if out is not None: seq += [out()]

        self.mlp = torch.nn.Sequential(*seq)

    def forward(self, x): 

        return self.mlp(x)

        