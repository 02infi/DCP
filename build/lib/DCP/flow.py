import numpy as numpy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.distributions.transforms as transform
# Imports for plotting
import matplotlib.pyplot as plt
import torch.optim as optim


class Flow(transform.Transform, nn.Module):
    
    def __init__(self):
        transform.Transform.__init__(self)
        nn.Module.__init__(self)
    
    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)
            
    def __hash__(self):
        return nn.Module.__hash__(self)




class RadialFlow(Flow):

    def __init__(self, dim,ref):
        super(RadialFlow, self).__init__()
        self.z0 = ref
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))
        self.dim = dim
        self.init_parameters()

    def _call(self, z):
        r = torch.norm(z - self.z0, dim=1).unsqueeze(1)
        h = 1 / (self.alpha + r)
        return z + (self.beta * h * (z - self.z0))

    def log_abs_det_jacobian(self, z):
        r = torch.norm(z - self.z0, dim=1).unsqueeze(1)
        h = 1 / (self.alpha + r)
        hp = - 1 / (self.alpha + r) ** 2
        bh = self.beta * h
        det_grad = ((1 + bh) ** self.dim - 1) * (1 + bh + self.beta * hp * r)
        return torch.log(det_grad.abs() + 1e-9)



class PlanarFlow(Flow):

    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.weight = nn.Parameter(torch.DoubleTensor(1, dim))
        self.scale = nn.Parameter(torch.DoubleTensor(1, dim))
        self.bias = nn.Parameter(torch.DoubleTensor(1))
        self.init_parameters()

    def _call(self, z):
        f_z = F.linear(z, self.weight, self.bias)
        return z + self.scale * torch.tanh(f_z)

    def log_abs_det_jacobian(self, z):
        f_z = F.linear(z, self.weight, self.bias)
        psi = (1 - torch.tanh(f_z) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return torch.log(det_grad.abs() + 1e-9)

class NormalizingFlow(nn.Module):

    def __init__(self, dim, flow_length,z_):
        super().__init__()
        biject = []
        for f in range(flow_length):
            #biject.append(RadialFlow(dim,z_))
            biject.append(PlanarFlow(dim))
        self.transforms = transform.ComposeTransform(biject)
        self.bijectors = nn.ModuleList(biject)
        self.log_det = []

    def forward(self, z):
        self.log_det = []
        # Applies series of flows
        for b in range(len(self.bijectors)):
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian(z))
            z = self.bijectors[b](z)
        return z, self.log_det

    def loss(self,density, zk, log_jacobians):
        sum_of_log_jacobians = sum(log_jacobians)
        #loss = (-sum_of_log_jacobians - density.log_prob(zk.detach().numpy())).mean()
        loss = (-sum_of_log_jacobians - torch.log(torch.Tensor(density(zk.detach().numpy().T)+1e-10))).mean()
        return loss


    def _train(self,cells,intial_density,terminal_density,iterations):
        self.start_density = intial_density
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)
        for it in range(iterations):
            samples = self.start_density.sample((cells,))
            zk, log_jacobians = self.forward(samples)
            optimizer.zero_grad()
            loss_v = self.loss(terminal_density, zk, log_jacobians)
            loss_v.backward()
            optimizer.step()
            scheduler.step()
            if (it % 10 == 0):
                print('Loss (it. %i) : %f'%(it, loss_v.item()))

    @torch.no_grad()
    def inference(self,z,z_mean):
        self.log_det = []
        self.z0 = z_mean
        # Applies series of flows
        for b in range(len(self.bijectors)):
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian(z))
            z = self.bijectors[b](z)
        return z       


