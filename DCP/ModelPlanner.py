import torch
from torch import optim
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule, LightningModule
from torch import Tensor


class Planner(LightningModule):

	def __init__(self,model,adata,workers,batch_size,**kwargs):

		super(Planner,self).__init__()
		self.model = model
		self.data = adata.X
		self.workers = workers
		self.batch_size = batch_size


	def forward(self,input:Tensor,**kwargs) -> Tensor:
		return self.model(input,**kwargs)


	def training_step(self,batch,batch_idx):
		train_results = self.forward(batch)
		train_loss = self.model._loss(*train_results)
		self.log("Performance",{loss_key:loss_value for loss_key, loss_value in train_loss.items()})
		return train_loss
		


#	def validation_step(self):
#		train_results = self.forward(batch)
#		val_loss = self.model.loss(*train_results)
#		self.log({loss_key: loss_value.item() for loss_key, loss_value in val_loss.items()})
#		return val_loss
		


	def configure_optimizers(self):
		optims = []
		optimizer = optim.Adam(self.model.parameters(),lr=0.0001)
		optims.append(optimizer)
		return optims


	def train_dataloader(self):
		return DataLoader(self.data,batch_size = self.batch_size,shuffle=True,num_workers=self.workers)

	