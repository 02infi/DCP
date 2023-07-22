import math
import torch
from .VAEtorch import VAEmodel
from .ModelPlanner import Planner
from .flow import NormalizingFlow
from .flow import PlanarFlow
from .flow import Flow
import pytorch_lightning
from pytorch_lightning import LightningDataModule, LightningModule
from anndata import AnnData
from pytorch_lightning import Trainer
from typing import List
from flow import * 
import scanpy as sc 
from scipy import sparse
import scipy.stats as ss

class DeepPredictor():
	def __init__(self, adata : AnnData, likelihood,latent_dim:int = 50, hidden_layers :List=None,flow_length:int = 32,workers:int=5,parameters=[1,0,1,0,0],batch_size:int = 100,log_sigma_coff:int = 1):
		super(DeepPredictor,self).__init__()
		self.adata = adata
		self.high_dim = np.shape(adata.X)[1]
		self.latent_dim = latent_dim
		self.batch_size = batch_size
		self.alpha,self.beta,self.gamma,self.lamda,self.epsilon = parameters
		self.sigma_coff = log_sigma_coff
		self.flow_length = flow_length
		self.likelihood = likelihood
		self.model = VAEmodel(alpha = self.alpha, beta = self.beta, gamma = self.gamma,
			lamda = self.lamda,epsilon=self.epsilon,likelihood = self.likelihood,sigma_coff = self.sigma_coff ,high_dim=self.high_dim,
			latent_dim=self.latent_dim,hidden_layers=hidden_layers,batch_size = self.batch_size)
		self.dVector = np.zeros(self.latent_dim)
		self.workers = workers
		if (likelihood == "nb"):
			self.library = np.sum(adata.X,axis=1).mean()
		else :
			self.library = 1

		
		
	def train(self,epochs):
		plan = Planner(self.model,self.adata,self.workers,self.batch_size)
		train_runner = Trainer(max_epochs=epochs)
		train_runner.fit(plan)

	def save_model(self,path):
		torch.save(self.model.state_dict(),path)


	def reload_model(self,path):
		self.model.load_state_dict(torch.load(path))
		self.model.eval()

	def diffVector(self,latent_a,latent_b):
		return np.mean(latent_b,axis=0) - np.mean(latent_a,axis=0)


	def balancing(self,adata,key,max_cells):
		key_variable = np.unique(adata.obs[key])
		index_arr = []
		cell_numbers = np.arange(adata.shape[0])
		for cells in key_variable:
			bool_index = np.array(adata.obs[key] == cells)
			index_ = np.random.choice(cell_numbers[bool_index],max_cells) 		
			index_arr.append(index_)
		return np.concatenate(index_arr)

	def Intialising_flows(self,z_):
		self.flow = NormalizingFlow(dim=self.latent_dim,flow_length=self.flow_length,z_=z_)



	def cellbalancer(self,adata_t1,adata_t2,key):
		max_cells_t1 = np.max(np.unique(adata_t1.obs[key].value_counts()))
		max_cells_t2 = np.max(np.unique(adata_t2.obs[key].value_counts()))
		total_cells_t1 = max_cells_t1 * len(adata_t1.obs[key].value_counts())
		total_cells_t2 = max_cells_t2 * len(adata_t2.obs[key].value_counts())

		if total_cells_t1 >= total_cells_t2:
			sampled_cells_t1 = int(np.round(np.divide(total_cells_t2,len(adata_t1.obs[key].value_counts()))))
			sampled_cells_t2 = max_cells_t2
		else:
			sampled_cells_t1 = max_cells_t1
			sampled_cells_t2 = int(np.round(np.divide(total_cells_t1,len(adata_t2.obs[key].value_counts()))))

		index_t1 = self.balancing(adata_t1,key,sampled_cells_t1)
		index_t2 = self.balancing(adata_t2,key,sampled_cells_t2)
		return index_t1,index_t2



	def runFlows(self,state_a,state_b,key,flow:bool = False,iterations = 100):
		adata_a = self.adata[self.adata.obs[key] == state_a,:]
		adata_b = self.adata[self.adata.obs[key] == state_b,:]
		index_a,index_b = self.cellbalancer(adata_a,adata_b,key)

		if sparse.issparse(self.adata.X):
			latent_a = self.model.to_latent(adata_a.X.A[index_a,:])
			latent_b = self.model.to_latent(adata_b.X.A[index_b,:])
			self.lVector = self.diffVector(adata_a.X.A[index_a,:],adata_b.X.A[index_b,:])

		else:
			latent_a = self.model.to_latent(adata_a.X[:])
			latent_b = self.model.to_latent(adata_b.X[:])
			self.lVector = self.diffVector(adata_a.X[index_a,:],adata_b.X[index_b,:])

		self.Intialising_flows(z_ = torch.tensor(np.mean(latent_b,axis=0),dtype=torch.float64))

		if not flow :
			self.dVector = self.diffVector(latent_a,latent_b) 
		else:
			self.dVector = 0
		latent_a_ = latent_a + self.dVector 

		kernel_a = ss.gaussian_kde(latent_a_.T)
		mean_a = np.mean(latent_a_.T,axis=1)
		covariance_a = kernel_a.covariance

		self.kernel_b = ss.gaussian_kde(latent_b.T)
		mean_b = np.mean(latent_b.T,axis=1)
		covariance_b = self.kernel_b .covariance

		self.IntialDistrib = distrib.MultivariateNormal(loc = torch.tensor(mean_a,dtype=torch.float64),
			covariance_matrix = torch.tensor(covariance_a,dtype=torch.float64))

		FinalDistrib = distrib.MultivariateNormal(loc = torch.tensor(mean_b,dtype=torch.float64),
			covariance_matrix = torch.tensor(covariance_b,dtype=torch.float64))



	def trainFlows(self,iterations=100,cell_numbers=500):
		self.flow._train(cells=cell_numbers,intial_density=self.IntialDistrib,terminal_density = self.kernel_b,iterations = iterations)




	def predict(self,adata_t,state_a,state_b,key):
		adata_test = adata_t[adata_t.obs[key] == state_a,:]
		adata_real = adata_t[adata_t.obs[key] == state_b,:]

		if sparse.issparse(self.adata.X):
			self.latent_test = self.model.to_latent(adata_test.X.A)
		else:
			self.latent_test = self.model.to_latent(adata_test.X)

		self.latent_predict = self.latent_test + self.dVector
		self.z_ = np.mean(self.latent_predict,axis=0)

		kernel_predict = ss.gaussian_kde(self.latent_predict.T)
		mean_predict = np.mean(self.latent_predict.T,axis=1)
		covariance_predict = kernel_predict.covariance

		latentDistrib = distrib.MultivariateNormal(loc = torch.tensor(mean_predict,dtype=torch.float64),
			covariance_matrix = torch.tensor(covariance_predict,dtype=torch.float64))
		latent_samples = latentDistrib.sample((adata_real.X.shape[0],))

		self.latent_predict = self.latent_predict.astype(np.float64)
		latent_predict = self.flow.inference(torch.tensor(latent_samples.detach().numpy()),self.z_)
		self.latent_predict = latent_predict.type(torch.FloatTensor)

		library = torch.log(torch.sum(torch.tensor(adata_test.X),dim=1,keepdim=True))

		input = torch.empty(np.shape(self.latent_predict)[0], 1)
		library_size = np.mean(np.sum(adata_test.X,axis=1))

		px_scale,theta,px_rate = self.model.latent_to(self.latent_predict,library=torch.ones_like(input))
		recontructed_data = library_size  * px_scale
		adata_predict = sc.AnnData(recontructed_data,obs={"cells": "Predicted data"}, var={"var_names":adata_t.var_names})
		adata_ = adata_predict.concatenate(adata_test,adata_real)
		return adata_,px_scale,theta,px_rate



	def predict_with_real_data(self,adata_t,state_a,state_b,key):
		adata_test = adata_t[adata_t.obs[key] == state_a,:]
		adata_real = adata_t[adata_t.obs[key] == state_b,:]

		if sparse.issparse(self.adata.X):
			self.latent_test = self.model.to_latent(adata_test.X.A)
			latent_real = self.model.to_latent(adata_real.X.A)
		else:
			self.latent_test = self.model.to_latent(adata_test.X)
			latent_real = self.model.to_latent(adata_real.X)

		self.latent_predict = self.latent_test + self.dVector

		kernel_predict = ss.gaussian_kde(self.latent_predict.T)
		mean_predict = np.mean(self.latent_predict.T,axis=1)
		covariance_predict = kernel_predict.covariance

		latentDistrib = distrib.MultivariateNormal(loc = torch.tensor(mean_predict,dtype=torch.float64),
			covariance_matrix = torch.tensor(covariance_predict,dtype=torch.float64))

		latent_samples = latentDistrib.sample((adata_real.X.shape[0],))
		self.latent_predict = self.latent_predict.astype(np.float64)

		#latent_predict = self.flow.inference(torch.tensor(latent_samples.detach().numpy()))
		self.latent_predict = self.flow.inference(torch.tensor(self.latent_predict))

		self.latent_predict = self.latent_predict.type(torch.FloatTensor)

		recontructed_data = self.model.latent_to(self.latent_predict)
		reconstructed_real_data = self.model.latent_to(torch.tensor(latent_real))

		adata_predict = sc.AnnData(recontructed_data,obs={"cells": "Predicted data"}, var={"var_names":adata_t.var_names})
		adata_real_ = sc.AnnData(reconstructed_real_data,obs={"cells": "Real data"}, var={"var_names":adata_t.var_names})

		adata_ = adata_predict.concatenate(adata_test,adata_real_)
		return adata_




	def scgen_predict(self,adata_t,state_a,state_b,key):
		adata_test = adata_t[adata_t.obs[key] == state_a,:]
		adata_real = adata_t[adata_t.obs[key] == state_b,:]

		if sparse.issparse(self.adata.X):
			latent_test = self.model.to_latent(adata_test.X.A)
		else:
			latent_test = self.model.to_latent(adata_test.X)

		latent_predict_scgen = latent_test + self.dVector

		latent_tensor = torch.from_numpy(latent_predict_scgen)

		recontructed_data = self.model.latent_to(latent_tensor)

		adata_predict = sc.AnnData(recontructed_data,obs={"cells": "Predicted data"}, var={"var_names":adata_test.var_names})
		adata_ = adata_predict.concatenate(adata_test,adata_real)
		return adata_



	def Vector_arthimetic(self,adata_t,state_a,state_b,key):
		adata_test = adata_t[adata_t.obs[key] == state_a,:]
		adata_real = adata_t[adata_t.obs[key] == state_b,:]
		max_size = max(adata_test.X.shape[0],adata_real.X.shape[0])
		index_test = np.random.choice(range(adata_test.X.shape[0]),size=max_size,replace=True)
		index_real = np.random.choice(range(adata_real.X.shape[0]),size=max_size,replace=True)

		adata_test = adata_test[index_test]

		if sparse.issparse(self.adata.X):
			recontructed_data = adata_test.X.A + self.lVector
		else:
			recontructed_data = adata_test.X + self.lVector
			
		recontructed_data_array = np.array(recontructed_data)
		#recontructed_data_array[recontructed_data_array < 0] = 0

		adata_predict = sc.AnnData(recontructed_data_array,obs={"cells": "Predicted data"}, var={"var_names":adata_t.var_names})

		adata_ = adata_predict.concatenate(adata_test,adata_real)
		return adata_



	def Simulations(self,adata_t,state_a,state_b,key,sim_num = 50):
		correlation_mean = []
		correlation_var = []
		correlation_slope_mean = []
		correlation_slope_var = [] 	
		for i in range(sim_num):
			adata_pred,px_scale,theta,px_rate = self.predict(adata_t,state_a,state_b,key)
			real_data = adata_pred[adata_pred.obs["cells"] == "Real data"]
			pred_data =  adata_pred[adata_pred.obs["cells"] == "Predicted data"] 
			slope, intercept, r, p_value, _err = ss.linregress(np.average(real_data.X,axis = 0), np.average(pred_data.X,axis = 0))
			slope_, intercept, r_, p_value, _err = ss.linregress(np.std(real_data.X,axis = 0), np.std(pred_data.X,axis = 0))
			correlation_slope_mean.append(slope)
			correlation_slope_var.append(slope_) 
			correlation_mean.append(r**2)
			correlation_var.append(r_**2)
			
		return correlation_mean,correlation_var,correlation_slope_mean,correlation_slope_var

	def Simulations_real_data(self,adata_t,state_a,state_b,key,sim_num = 50):
		correlation_mean = []
		correlation_var = []
		correlation_slope_mean = []
		correlation_slope_var = [] 	
		for i in range(sim_num):
			adata_pred, = self.predict_with_real_data(adata_t,state_a,state_b,key)
			real_data = adata_pred[adata_pred.obs["cells"] == "Real data"]
			pred_data =  adata_pred[adata_pred.obs["cells"] == "Predicted data"] 
			slope, intercept, r, p_value, _err = ss.linregress(np.average(real_data.X,axis = 0), np.average(pred_data.X,axis = 0))
			slope_, intercept, r_, p_value, _err = ss.linregress(np.std(real_data.X,axis = 0), np.std(pred_data.X,axis = 0))
			correlation_slope_mean.append(slope)
			correlation_slope_var.append(slope_) 
			correlation_mean.append(r**2)
			correlation_var.append(r_**2)
		return correlation_mean,correlation_var,correlation_slope_mean,correlation_slope_var


	def FeatureInterpretability(vector):
		vector = torch.from_numpy(vector)
		recontructed_data = model.latent_to(vector)
		return recontructed_data

	


		





		










