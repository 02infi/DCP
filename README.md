# Deep cell predictor (DCP)
Deep cell predictor (DCP) is a deep learning approach that explicitly models changes in transcriptional variance using a combination of variational autoencoders and normalizing flows.   

DCP is described in   
Inference of differentiation trajectories by transfer learning across biological processes
Gaurav Jumde,Bastiaan Spanjaard, Jan Philipp Junker*

# Installation
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

# Example code 
The prediction analysis from Jumde et al. are as follows:   
[Zebrafish](python_notebooks/zebrafish)   
[Hematopoiesis](python_notebooks/hematopoiesis/With_all_genes)
