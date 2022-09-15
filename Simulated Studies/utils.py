import numpy as np
import torch
import torch.nn as nn


def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128, cubic=None):
    """Generate samples from a correlated Gaussian distribution."""
    x, eps = torch.chunk(torch.randn(batch_size, 2 * dim), 2, dim=1)
    y = rho * x + torch.sqrt(torch.tensor(1. - rho**2).float()) * eps

    if cubic is not None:
        y = y ** 3

    return x, y


def rho_to_mi(dim, rho):
    """Obtain the ground truth mutual information from rho."""
    return -0.5 * np.log(1 - rho**2) * dim


def mi_to_rho(dim, mi):
    """Obtain the rho for Gaussian give ground truth mutual information."""
    return np.sqrt(1 - np.exp(-2.0 / dim * mi))

def rho_to_upper(rho,dim):
    xx = 1 - rho*rho
    result = -1 + 0.5 * np.log(xx) + 1/xx
    
    return dim*result


def mi_schedule(n_iter):
    """Generate schedule for increasing correlation over time."""
    mis = np.round(np.linspace(0.5, 5.5 - 1e-9, n_iter)) * 2.0
    return mis.astype(np.float32)


def mlp(dim, hidden_dim, output_dim, layers, activation):
    """Create a mlp from the configurations."""
    activation = {
        'relu': nn.ReLU
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)


class SeparableCritic(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self, dim, hidden_dim, embed_dim, layers, activation, **extra_kwargs):
        super(SeparableCritic, self).__init__()
        self._g = mlp(dim, hidden_dim, embed_dim, layers, activation)
        self._h = mlp(dim, hidden_dim, embed_dim, layers, activation)

    def forward(self, x, y):
        scores = torch.matmul(self._h(y), self._g(x).t())
        return scores

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


class NCECritic(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self, dim, hidden_dim, embed_dim, layers, activation, **extra_kwargs):
        super(NCECritic, self).__init__()
        self._g = mlp(dim, hidden_dim, embed_dim, layers, activation)
        self._h = mlp(dim, hidden_dim, embed_dim, layers, activation)

    def forward(self, x, y):
        out_1 = self._g(x)
        out_2 = self._h(y)
        temperature = 1
        batch_size = x.size(0)
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = get_negative_mask(batch_size)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        # if debiased:
        #     N = batch_size * 2 - 2
        #     Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
        #     # constrain (optional)
        #     Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        # else:
        Ng = neg.sum(dim=-1)

        # contrastive loss
        mi = ( torch.log(pos / (pos + Ng) )).mean() + torch.tensor(x.size(0)).float().log()
        
        return mi

class DNCECritic(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self, dim, hidden_dim, embed_dim, layers, activation, **extra_kwargs):
        super(DNCECritic, self).__init__()
        self._g = mlp(dim, hidden_dim, embed_dim, layers, activation)
        self._h = mlp(dim, hidden_dim, embed_dim, layers, activation)

    def forward(self, x, y):
        out_1 = self._g(x)
        out_2 = self._h(y)
        temperature = 1
        batch_size = x.size(0)
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = get_negative_mask(batch_size)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        # if debiased:
        tau_plus = 0.1
        N = batch_size * 2 - 2
        Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))

        # contrastive loss
        mi = ( torch.log(pos / (pos + Ng) )).mean() + torch.tensor(x.size(0)).float().log()
        
        return mi

class ConcatCritic(nn.Module):
    """Concat critic, where we concat the inputs and use one MLP to output the value."""

    def __init__(self, dim, hidden_dim, layers, activation, **extra_kwargs):
        super(ConcatCritic, self).__init__()
        # output is scalar score
        self._f = mlp(dim * 2, hidden_dim, 1, layers, activation)

    def forward(self, x, y):
        batch_size = x.size(0)
        # Tile all possible combinations of x and y
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [
                                 batch_size * batch_size, -1])
        # Compute scores for each x_i, y_j pair.
        scores = self._f(xy_pairs)
        return torch.reshape(scores, [batch_size, batch_size]).t()

class CLUB_layers(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    
    def __init__(self, dim, hidden_dim,layers,activation = 'relu'):
        super(CLUB_layers, self).__init__()
        x_dim = dim
        y_dim = dim
        hidden_size = hidden_dim//2

        activation = {
            'relu': nn.ReLU
        }[activation]

        seq_mu = [nn.Linear(x_dim, hidden_size), activation()]
        for _ in range(layers):
            seq_mu += [nn.Linear(hidden_size, hidden_size), activation()]
        seq_mu += [nn.Linear(hidden_size, y_dim)]

        seq_var = [nn.Linear(x_dim, hidden_size), activation()]
        for _ in range(layers):
            seq_var += [nn.Linear(hidden_size, hidden_size), activation()]
        seq_var += [nn.Linear(hidden_size, y_dim)]

        self.p_mu = nn.Sequential(*seq_mu)
       
        self.p_logvar = nn.Sequential(*seq_var)

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        loglikeli = (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)

        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        upper = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
      
        return loglikeli + (-loglikeli + upper).detach()

class L1OutUB(nn.Module):  # naive upper bound
    def __init__(self, dim, hidden_size):

        super(L1OutUB, self).__init__()
        x_dim = dim
        y_dim = dim
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples): 
        batch_size = y_samples.shape[0]
        mu, logvar = self.get_mu_logvar(x_samples)

        positive = (- (mu - y_samples)**2 /2./logvar.exp() - logvar/2.).sum(dim = -1) #[nsample]

        mu_1 = mu.unsqueeze(1)          # [nsample,1,dim]
        logvar_1 = logvar.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)            # [1,nsample,dim]
        all_probs =  (- (y_samples_1 - mu_1)**2/2./logvar_1.exp()- logvar_1/2.).sum(dim = -1)  #[nsample, nsample]

        diag_mask =  torch.ones([batch_size]).diag().unsqueeze(-1) * (-20.)
        negative = log_sum_exp(all_probs + diag_mask,dim=0) - np.log(batch_size-1.) #[nsample]
      
        upper  = (positive - negative).mean()
        loglikeli = (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)

        return loglikeli + (-loglikeli + upper).detach()

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

class VarUB(nn.Module):  #    variational upper bound
    def __init__(self, dim, hidden_size):
        super(VarUB, self).__init__()
        x_dim = dim
        y_dim = dim
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
            
    def forward(self, x_samples, y_samples): #[nsample, 1]
        mu, logvar = self.get_mu_logvar(x_samples)
        upper = 1./2.*(mu**2 + logvar.exp() - 1. - logvar).mean()
        loglikeli = (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
        return loglikeli + (-loglikeli + upper).detach()
        
    # def loglikeli(self, x_samples, y_samples):
    #     mu, logvar = self.get_mu_logvar(x_samples)
    #     return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)

    # def learning_loss(self, x_samples, y_samples):
    #     return - self.loglikeli(x_samples, y_samples)






def log_prob_gaussian(x):
    return torch.sum(torch.distributions.Normal(0., 1.).log_prob(x), -1)