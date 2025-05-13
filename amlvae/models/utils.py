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
    elif nonlin == 'mish':
        return torch.nn.Mish
    elif nonlin == 'swish':
        return torch.nn.Swish
    elif nonlin == 'gelu':
        return torch.nn.GELU
    elif nonlin == 'none':
        return torch.nn.Identity
    else:
        raise ValueError(f"Activation function {nonlin} not recognized")
    
def get_norm(norm): 

    if norm == 'batch': 
        return torch.nn.BatchNorm1d
    elif norm == 'layer':
        return torch.nn.LayerNorm
    elif norm == 'none': 
        return lambda x: torch.nn.Identity()
    else:
        raise ValueError(f"Normalization layer {norm} not recognized")