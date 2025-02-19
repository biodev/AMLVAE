import torch

def get_nonlin(nonlin): 

    if nonlin == 'relu': 
        return torch.nn.ReLU
    elif nonlin == 'elu':
        return torch.nn.ELU
    elif nonlin == 'tanh':
        return torch.nn.Tanh
    elif nonlin == 'sigmoid':
        return torch.nn.Sigmoid
    else:
        raise ValueError(f"Activation function {nonlin} not recognized")
    
def get_norm(norm): 

    if norm == 'batch': 
        return torch.nn.BatchNorm1d
    elif norm == 'layer':
        return torch.nn.LayerNorm
    else:
        raise ValueError(f"Normalization layer {norm} not recognized")