## Deep cell predictor (DCP)
Deep cell predictor (DCP) is a deep learning approach that explicitly models changes in transcriptional variance using a combination of variational autoencoders and normalizing flows.   

## Installation
To install Deep cell predictor (DCP) via pip : 

```bash
pip install deepcellpredictor
```
Alterantively, one can install DCP master branch directly from github :
```bash
python -m pip install git+https://github.com/02infi/DCP.git
```
To uninstall the package,
```bash
pip uninstall deepcellpredictor
```

## Tutorial
Getting started with simple and quick demonstration of DCP workflow as follows.  

```python
# Importing necessary libraries
import DCP
from DCP import deep_predictor


# Read the data and define training and test data
Adata = sc.read()
Training_data = Adata[Adata.obs[“”] == “training label”]
Test_data = Adata[Adata.obs[“”] == “test label”]


# Initialize the DCP object
DCP_object = deep_predictor.DeepPredictor(training_data,
                                          latent_dim=100,
                                          hidden_layers=[800,800],
                                          workers=8,
                                          parameters=[0,0,0,1,1],
                                          likelihood="nb",
                                          batch_size=100)


# Training the nb-mmdVAE 
DCP_object.train(epochs=number of epochs)


# Initialise and run the flows
DCP_object.runFlows(Timepoint_1,Timepoint_2,"Timelabel")
DCP_object.trainFlows(iterations= number of iterations)


Predicted_object,px_scale,theta,px_rate = DCP_object.predict (Test_data,
                                                              Timepoint_1,
                                                              Timepoint_2,
                                                              "Timelabel")


# Plot the correlation plots 
Plotting_Correlation_genes(Predicted_object)
Plotting_Correlation_var(Predicted_object)


# Calculate and PCA plots
Calculating_PCA(Predicted_object)
Plotting_PCA(Predicted_object, parameters = …)

```


## Example code 
DCP Prediction analysis over single cell data comparing early development in zebrafish and mouse with Mouse and Hydra stem cell differenation systems  
[Zebrafish](https://nbviewer.org/github/02infi/DCP/tree/main/python_notebooks/zebrafish/)   
[Hematopoiesis](https://nbviewer.org/github/02infi/DCP/tree/main/python_notebooks/hematopoiesis/With_all_genes/)  
[Mouse](https://nbviewer.org/github/02infi/DCP/tree/main/python_notebooks/mouse/)  
[Hydra](https://nbviewer.org/github/02infi/DCP/tree/main/python_notebooks/hydra/)  

The detailed description of analysis from Jumde et al. 2023 paper is given
[here](https://nbviewer.org/github/02infi/DCP/tree/main/python_notebooks/figures/)

## Questions 
If you have any question related to installation or running DCP, please open the github issue [here](https://github.com/02infi/DCP/issues/new)
 
