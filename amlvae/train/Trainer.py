

import torch 

class Trainer: 

    def __init__(self, model):

        self.model = model

    def epoch(self, X, optim, batch_size=128, beta=1, eval=False, device='cpu'):
        '''
        run one epoch 
        '''
        
        if eval:
            self.model.eval()
        else:
            self.model.train()

        epoch_loss = 0
        if eval: xs = [] ; xhats = []
        with torch.set_grad_enabled(not eval):

            if eval: 
                ixs_splits = torch.split(torch.arange(len(X)), batch_size)
            else:
                ixs_splits = torch.split(torch.randperm(len(X)), batch_size)


            for ixs in ixs_splits:
                
                x = X[ixs]
                x = x.to(device)

                if not eval: 
                    optim.zero_grad()

                out = self.model(x)
                loss = self.model.loss(x, beta=beta, **out)
                
                if not eval: 
                    loss.backward()
                    optim.step()

                epoch_loss += loss.item()
                if eval: 
                    xs.append(x)
                    xhats.append(out['xhat'])

        epoch_loss /= len(ixs_splits)

        if eval: 
            xs = torch.cat(xs, dim=0)
            xhats = torch.cat(xhats, dim=0)
            metrics = self.model.eval_(xs, xhats)
        else: 
            metrics = {}

        return epoch_loss, metrics 
    
    def train(self, X_train, X_val, epochs, batch_size=128, beta=1, lr=1e-3, verbose=True):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.to(device)

        best_state = None 
        best_loss = float('inf')

        for epoch in range(epochs):
            train_loss, _ = self.epoch(X_train, optim, batch_size, beta, eval=False, device=device)
            val_loss, val_metrics = self.epoch(X_val, None, batch_size, beta, eval=True, device=device)

            if verbose:
                val_str = ' | '.join([f'{k}: {v:.4f}' for k, v in val_metrics.items()])
                print(f'epoch: {epoch+1}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f} ||>>val>>| {val_str}', end='\r')

            if val_metrics['MSE'] < best_loss:
                best_loss = val_metrics['MSE']
                best_state = self.model.state_dict()

        self.best_state = best_state

    def get_best_model(self): 
        model = self.model 
        model.load_state_dict(self.best_state)
        model.cpu()
        return model