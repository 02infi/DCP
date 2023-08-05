import torch
import scvi
from torch import nn
from torch.nn import functional as F
from typing import List
from torch import Tensor
from torch.distributions import Normal
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
import numpy as np
from torch.autograd import Variable

class VAEmodel(nn.Module):

    def __init__(self,alpha,beta,gamma,lamda,epsilon,likelihood,sigma_coff,high_dim: int,latent_dim: int,hidden_layers: List = None,batch_size:int = 32,**kwargs):

        super(VAEmodel, self).__init__()

        if latent_dim is None:
            self.latent_dim = 100
        else :
        	self.latent_dim = latent_dim

        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma
        self.lamda = lamda
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.log_sigma_coff = sigma_coff
        self.likelihood = likelihood
        layer_modules = []
        if hidden_layers is None:
            hidden_layers = [800,800]
        self.likelihood = likelihood
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        bias = False
        drop_rate = 0.1 
        n_in = high_dim
        # Encoder 
        for layer in hidden_layers:
        	layer_modules.append(
        		nn.Sequential(
        			nn.Linear(n_in,out_features = layer,bias=bias),
        			nn.BatchNorm1d(layer),
        			nn.LeakyReLU(negative_slope=0.3),
        			nn.Dropout(p=drop_rate)
        			)
        		)
        	n_in = layer 

        self.enc = nn.Sequential(*layer_modules)
        self.mean_layer = nn.Linear(hidden_layers[-1],latent_dim,bias=bias)
        self.var_layer = nn.Linear(hidden_layers[-1],latent_dim,bias=bias)

        # Decoder 
        layer_modules = []

        #self.latent_layer = nn.Linear(latent_dim,hidden_layers[-1],bias=bias)
        hidden_layers.append(self.latent_dim)
        hidden_layers.reverse()
        #hidden_layers.append(high_dim)


        #n_in = latent_dim
        for i in range(len(hidden_layers)-1):
            layer_modules.append(
                nn.Sequential(
                    nn.Linear(hidden_layers[i],out_features = hidden_layers[i+1],bias=bias),
                    nn.BatchNorm1d(hidden_layers[i+1]),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.Dropout(p=drop_rate)
                    )
                )
            n_in = hidden_layers[i+1] 


        if (likelihood == "nb"):
            self.dec = nn.Sequential(*layer_modules)
        else :
            layer_modules.append(
                nn.Sequential(
                    nn.Linear(n_in,out_features = high_dim,bias=bias),
                    nn.ReLU(),
                    ))
            self.dec = nn.Sequential(*layer_modules)

        #self.dec = nn.Sequential(*layer_modules)

        self.log_theta =  torch.nn.Parameter(torch.randn(high_dim))
        self.px_scale_decoder = nn.Sequential(nn.Linear(n_in, high_dim), nn.Softmax(dim=-1))
        #self.log_theta =  Variable(torch.randn(high_dim),requires_grad=True)
    
    def encoder(self, input: Tensor) -> List[Tensor]:
        l = self.enc(input)
        l_mu = self.mean_layer(l)
        l_var = self.var_layer(l)
        return [l_mu,l_var]

    def decoder(self, z: Tensor,library) -> List[Tensor]:
        theta = 0
        px_rate = 0
        if (self.likelihood =="nb"):
            px = self.dec(z)
            px_scale = self.px_scale_decoder(px)            
            px_rate = library * px_scale
            #px_r = torch.exp(self.px_r)
            theta = torch.exp(self.log_theta)
        else :
            px_scale = self.dec(z)
        return [px_scale,theta,px_rate]


    def reparameterize(self,l_mu: Tensor,l_var : Tensor) -> Tensor:
        l_std = torch.exp(l_var / 2)
        l_eps = torch.randn_like(l_std)
        return l_eps * l_std + l_mu

    @torch.no_grad()     
    def sample_(self,n_samples:int, current_device: int) -> Tensor:
        z = torch.randn(n_samples,self.latent_dim)
        z = z.to(current_device)
        z_ = self.decoder(z)
        return z_


    def reconstruction_loss(self,x,x_) -> torch.Tensor:
        loss = ((x - x_) ** 2).sum(dim=1)
        return loss

    @torch.no_grad()    
    def to_latent(self,input_data: Tensor) -> np.ndarray:
        x_ = torch.from_numpy(input_data)
        l_mu,l_var = self.encoder(x_.to(self.device))
        z = self.reparameterize(l_mu,l_var)
        return z.detach().numpy()

    @torch.no_grad()
    def latent_to(self, z: Tensor,library) -> np.ndarray:
        #z = torch.from_numpy(input_data)
        #z = torch.tensor(z,dtype=torch.float64)
        px_scale,theta,px_rate = self.decoder(z.to(self.device),library)
        return px_scale.detach().numpy(),theta.detach().numpy(),px_rate.detach().numpy()
        

    def _reparameterize(self,l_mu: Tensor, l_var: Tensor) -> Tensor:
        l_std = torch.exp(l_var/2)
        _q = torch.distributions.Normal(l_mu,l_std)
        _z = _q.rsample()
        return _z,l_std 

    def _kl_div(self,_z,l_mu,l_std):

        p = torch.distributions.Normal(torch.zeros_like(l_mu), torch.ones_like(l_std))
        q = torch.distributions.Normal(l_mu,l_std)

        log_qz = q.log_prob(_z)
        log_pz = p.log_prob(_z)

        kl = (log_qz - log_pz)
        kl = kl.sum(1)
        return kl 


    def _computing_kernel(self, x: Tensor, y: Tensor) -> Tensor:
        x_cells = x.size(0)
        y_cells = y.size(0)
        latent_dim = x.size(1)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        rect_x = x.expand(x_cells,y_cells,latent_dim)
        rect_y = y.expand(x_cells,y_cells,latent_dim)
        input_kernel = (rect_x - rect_y).pow(2).mean(2)/float(latent_dim)
        return torch.exp(-input_kernel)



    def _compute_mmd(self,x: Tensor) -> Tensor:
        y = torch.randn_like(x)
        x_kernel = self._computing_kernel(x, x)
        xy_kernel = self._computing_kernel(x, y)
        y_kernel = self._computing_kernel(y, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        return mmd



    def forward(self,input_data:Tensor) -> List[Tensor]:
        l_mu,l_var = self.encoder(input_data)
        z,l_std = self._reparameterize(l_mu,l_var)
        library = torch.log(torch.sum(input_data,dim=1,keepdim=True))
        px_scale,theta,px_rate = self.decoder(z,library)
        return [input_data,l_mu,l_var,z,px_scale,theta,px_rate]


    def softclip(self,tensor, min):
        result_tensor = min + F.softplus(tensor - min)
        return result_tensor


    def gaussian_nll(self,mu, log_sigma, x):
        return  0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) +  log_sigma 
        #return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + self.log_sigma_coff * log_sigma + 0.5 * np.log(2 * np.pi)


    def reconstruction_loss_sigma(self,x,x_) -> torch.Tensor:
        #loss = ((x - x_) ** 2).sum(dim=1)
        log_sigma = ((x - x_) ** 2).mean([0,1], keepdim=True).sqrt().log()
        #log_sigma = self.softclip(log_sigma, -0.5)
        loss = self.gaussian_nll(x_,log_sigma,x).mean()
        return loss, log_sigma


    def _loss(self,*args) -> dict:
        input_data = args[0]
        l_mu = args[1]
        l_var = args[2]
        _z = args[3]
        px_scale = args[4] # px         
        theta = args[5]
        px_rate = args[6]
        reconstructed_data = px_scale
        log_likelihood_ = -NegativeBinomial(mu=px_rate, theta=theta).log_prob(input_data).sum(dim=-1).mean()
        reconstructed_loss = self.reconstruction_loss(input_data,reconstructed_data).mean()
        reconstructed_loss_sigma , log_sigma = self.reconstruction_loss_sigma(input_data,reconstructed_data)
        Kl_loss = 0.5 * torch.sum(torch.exp(l_var) + l_mu**2 - 1 - l_var)
        Mmd_loss = self._compute_mmd(_z).mean()
        total_loss = (self.alpha * reconstructed_loss + self.beta * reconstructed_loss_sigma + self.gamma * Kl_loss + self.lamda * Mmd_loss + self.epsilon * log_likelihood_)
        return {'loss' : total_loss,"MSE_loss": reconstructed_loss,"NB_likelihood": log_likelihood_,'Sigma_Reconstruction_loss' : reconstructed_loss_sigma, 'Kl_divergence_loss':Kl_loss, 'Mmd_loss':Mmd_loss, "Log_sigma": log_sigma}










        

    















        
        	








