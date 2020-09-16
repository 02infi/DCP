import scgen
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import rcParams
from scipy.sparse import csr_matrix
import scanpy as sc
from anndata import AnnData
import h5py, logging
sc.set_figure_params(dpi=500, color_map='viridis')  # low dpi (dots per inch) yields small inline figures
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
#sc.logging.print_versions() 
sc.settings.set_figure_params(dpi=500)
#pl.rcParams['figure.facecolor'] = 'white'
import tensorflow as tf
import numpy
from scipy import sparse
import sklearn 
import anndata
from anndata import AnnData
import matplotlib.pyplot as plt
from sklearn import decomposition


import os 
from multiprocessing import Pool
import multiprocessing
import seaborn as sns
from scipy import sparse

from scipy import stats

from scipy.spatial import distance





def corrfunc(x, y, **kws):
    slope, intercept, r, p_value, _err = stats.linregress(x, y)
    ax = plt.gca()
    ax.annotate("$R^2$ = {:.2f}".format(r**2),
                xy=(.1, .9), xycoords=ax.transAxes)


def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)



def Correlation_plots(scanpy_obj):
    timepoints = np.unique(scanpy_obj.obs['HPF'])
    timepoints = timepoints[1:]
    data = []
    for i in range(len(timepoints)):
        data.append(np.average(scanpy_obj[(scanpy_obj.obs['HPF'] == timepoints[i])].X,axis = 0))
        
    scanpy_obj_df = pd.DataFrame(np.array(data).T,columns = timepoints)
    g = sns.PairGrid(scanpy_obj_df, palette=["red"])
    g.map_upper(hide_current_axis)
    g.map_diag(sns.distplot)
    g.map_lower(plt.scatter, cmap="Blues_d",s =10)
    g.map_lower(corrfunc)




def balancer(adata, cell_type_key="cell_type", condition_key="condition"):
    """
        Makes cell type population equal.

        # Parameters
        adata: `~anndata.AnnData`
            Annotated data matrix.

        # Returns
            balanced_data: `~anndata.AnnData`
                Equal cell type population Annotated data matrix.

        # Example
        ```python
        import scgen
        import anndata
        train_data = anndata.read("./train_kang.h5ad")
        train_ctrl = train_data[train_data.obs["condition"] == "control", :]
        train_ctrl = balancer(train_ctrl)
        ```
    """
    class_names = np.unique(adata.obs[cell_type_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = adata.copy()[adata.obs[cell_type_key] == cls].shape[0]
    max_number = np.max(list(class_pop.values()))
    all_data_x = []
    all_data_label = []
    all_data_condition = []
    for cls in class_names:
        temp = adata.copy()[adata.obs[cell_type_key] == cls]
        index = np.random.choice(range(len(temp)), max_number)
        if sparse.issparse(temp.X):
            temp_x = temp.X.A[index]
        else:
            temp_x = temp.X[index]
        all_data_x.append(temp_x)
        temp_ct = np.repeat(cls, max_number)
        all_data_label.append(temp_ct)
        temp_cc = np.repeat(np.unique(temp.obs[condition_key]), max_number)
        all_data_condition.append(temp_cc)
    balanced_data = anndata.AnnData(np.concatenate(all_data_x))
    balanced_data.obs[cell_type_key] = np.concatenate(all_data_label)
    balanced_data.obs[condition_key] = np.concatenate(all_data_label)
    class_names = np.unique(balanced_data.obs[cell_type_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = len(balanced_data[balanced_data.obs[cell_type_key] == cls])
    return balanced_data





def Linear_interpolations(adata, conditions, cell_type_key, condition_key, adata_to_predict=None, celltype_to_predict=None,
            obs_key="all", biased=False):
    """
    """
    if obs_key == "all":
        ctrl_x = adata[adata.obs[condition_key] == conditions["ctrl"], :]
        stim_x = adata[adata.obs[condition_key] == conditions["stim"], :]
        if not biased:
            ctrl_x = balancer(ctrl_x, cell_type_key=cell_type_key, condition_key=condition_key)
            stim_x = balancer(stim_x, cell_type_key=cell_type_key, condition_key=condition_key)
    else:
        key = list(obs_key.keys())[0]
        values = obs_key[key]
        subset = adata[adata.obs[key].isin(values)]
        ctrl_x = subset[subset.obs[condition_key] == conditions["ctrl"], :]
        stim_x = subset[subset.obs[condition_key] == conditions["stim"], :]
        if len(values) > 1 and not biased:
            ctrl_x = balancer(ctrl_x, cell_type_key=cell_type_key, condition_key=condition_key)
            stim_x = balancer(stim_x, cell_type_key=cell_type_key, condition_key=condition_key)
    if celltype_to_predict is not None and adata_to_predict is not None:
        raise Exception("Please provide either a cell type or adata not both!")
    if celltype_to_predict is None and adata_to_predict is None:
        raise Exception("Please provide a cell type name or adata for your unperturbed cells")
    if celltype_to_predict is not None:
        ctrl_pred = extractor(adata, celltype_to_predict, conditions, cell_type_key, condition_key)[1]
    else:
        ctrl_pred = adata_to_predict
    if not biased:
        eq = min(ctrl_x.X.shape[0], stim_x.X.shape[0])
        cd_ind = numpy.random.choice(range(ctrl_x.shape[0]), size=eq, replace=False)
        stim_ind = numpy.random.choice(range(stim_x.shape[0]), size=eq, replace=False)
    else:
        cd_ind = numpy.random.choice(range(ctrl_x.shape[0]), size=ctrl_x.shape[0], replace=False)
        stim_ind = numpy.random.choice(range(stim_x.shape[0]), size=stim_x.shape[0], replace=False)
    if sparse.issparse(ctrl_x.X) and sparse.issparse(stim_x.X):
        latent_ctrl = np.average(ctrl_x.X[cd_ind, :],axis = 0)
        latent_sim = np.average(stim_x.X[stim_ind, :],axis = 0)
    else:
        latent_ctrl = np.average(ctrl_x.X[cd_ind, :],axis = 0)
        latent_sim = np.average(stim_x.X[stim_ind, :],axis = 0)
    delta = latent_sim - latent_ctrl

    latent_cd = ctrl_pred.X
    stim_pred = delta + latent_cd
    return stim_pred, delta




 
def to_PCA_space(data,weights):
    return np.dot(data,weights)



def PCA_interpolations(adata, conditions, cell_type_key, condition_key, adata_to_predict=None, celltype_to_predict=None,obs_key="all", biased=False, pc_components = 100):

    PCA_training_data = adata
    #mu = np.mean(PCA_training_data, axis=0)
    sc.tl.pca(PCA_training_data,n_comps=100,return_info=True)
    #pca = sklearn.decomposition.SparsePCA(n_components=pc_components)
    #pca.fit(PCA_training_data.toarray())
    #pca_Loadings = np.dot(PCA_training_data.toarray(), pca.components_.T) # transform
    pca_Loadings = PCA_training_data.varm["PCs"]


    if obs_key == "all":
        ctrl_x = adata[adata.obs[condition_key] == conditions["ctrl"], :]
        stim_x = adata[adata.obs[condition_key] == conditions["stim"], :]
        if not biased:
            ctrl_x = balancer(ctrl_x, cell_type_key=cell_type_key, condition_key=condition_key)
            stim_x = balancer(stim_x, cell_type_key=cell_type_key, condition_key=condition_key)
    else:
        key = list(obs_key.keys())[0]
        values = obs_key[key]
        subset = adata[adata.obs[key].isin(values)]
        ctrl_x = subset[subset.obs[condition_key] == conditions["ctrl"], :]
        stim_x = subset[subset.obs[condition_key] == conditions["stim"], :]
        if len(values) > 1 and not biased:
            ctrl_x = balancer(ctrl_x, cell_type_key=cell_type_key, condition_key=condition_key)
            stim_x = balancer(stim_x, cell_type_key=cell_type_key, condition_key=condition_key)
    if celltype_to_predict is not None and adata_to_predict is not None:
        raise Exception("Please provide either a cell type or adata not both!")
    if celltype_to_predict is None and adata_to_predict is None:
        raise Exception("Please provide a cell type name or adata for your unperturbed cells")
    if celltype_to_predict is not None:
        ctrl_pred = extractor(adata, celltype_to_predict, conditions, cell_type_key, condition_key)[1]
    else:
        ctrl_pred = adata_to_predict
    if not biased:
        eq = min(ctrl_x.X.shape[0], stim_x.X.shape[0])
        cd_ind = numpy.random.choice(range(ctrl_x.shape[0]), size=eq, replace=False)
        stim_ind = numpy.random.choice(range(stim_x.shape[0]), size=eq, replace=False)
    else:
        cd_ind = numpy.random.choice(range(ctrl_x.shape[0]), size=ctrl_x.shape[0], replace=False)
        stim_ind = numpy.random.choice(range(stim_x.shape[0]), size=stim_x.shape[0], replace=False)


    if sparse.issparse(ctrl_x.X) and sparse.issparse(stim_x.X):
        latent_ctrl = np.average(to_PCA_space(ctrl_x[cd_ind,:].X,pca_Loadings),axis = 0)
        latent_sim = np.average(to_PCA_space(stim_x[stim_ind,:].X,pca_Loadings),axis = 0)
    else:
        latent_ctrl = np.average(to_PCA_space(ctrl_x[cd_ind,:].X,pca_Loadings),axis = 0)
        latent_sim = np.average(to_PCA_space(stim_x[stim_ind,:].X,pca_Loadings),axis = 0)
    delta = latent_sim - latent_ctrl

    if sparse.issparse(ctrl_pred.X):

        PCA_test_data = ctrl_pred.X
        latent_cd = to_PCA_space(PCA_test_data,pca_Loadings)
    else:

        PCA_test_data = ctrl_pred.X
        latent_cd = to_PCA_space(PCA_test_data,pca_Loadings)
        
    stim_pred = delta + latent_cd
    predicted_cells = np.dot(pca_Loadings,stim_pred.T)
    
    gene_expression_delta = np.dot(pca_Loadings,delta.T)
    predicted_cells_expression_data = np.add(PCA_test_data,gene_expression_delta)    
    return predicted_cells, delta,predicted_cells_expression_data




from scipy.spatial import distance
import seaborn as sns

def compute_probs(data, n): 
    h, e = np.histogram(data, bins = n)
    p = h/np.array(data).shape[0]
    return e, p


def get_probs(list_of_tuples): 
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q


def compute_js_divergence(real_sample, pred_sample, n_bins=50): 
    
    ##Computes the JS Divergence using the support intersection between two different samples
    
    e, p = compute_probs(np.array(real_sample), n= "auto")
    _, q = compute_probs(np.array(pred_sample), n=e)    
    return distance.jensenshannon(p, q)
    
def corrfunc(x, y, **kws):
    slope, intercept, r, p_value, _err = stats.linregress(x, y)
    ax = plt.gca()
    ax.annotate("$R^2$ = {:.2f}".format(r**2),
                xy=(.1, .9), xycoords=ax.transAxes)

def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)

    
def Correlation_plots(scanpy_obj):
    timepoints = np.unique(scanpy_obj.obs['HPF'])
    timepoints = timepoints[1:]
    data = []
    for i in range(len(timepoints)):
        data.append(np.average(scanpy_obj[(scanpy_obj.obs['HPF'] == timepoints[i])].X,axis = 0))
        
    scanpy_obj_df = pd.DataFrame(np.array(data).T,columns = timepoints)
    g = sns.PairGrid(scanpy_obj_df, palette=["red"])
    g.map_upper(hide_current_axis)
    g.map_diag(sns.distplot)
    g.map_lower(plt.scatter, cmap="Blues_d",s =10)
    g.map_lower(corrfunc)

    

def sample_gene(real_sample, pred_sample, n_bins=10):

    gene_array = []

    e, p = compute_probs(real_sample, n=n_bins)
    _, q = compute_probs(pred_sample, n=e)

    for i in range(0,1000):
        p_index = np.random.choice(10, size=5, replace=True)
        q_index = np.random.choice(10, size=5, replace=True)
        gene_array.append(distance.jensenshannon(p[p_index],q[q_index]))

    return gene_array



def multiprocessing_JS_function(first_Obj,Last_Obj):
    job_args = zip(np.array(first_Obj.X.T),np.array(Last_Obj.X.T)) 
    p = Pool(processes=10)
    output = p.starmap(compute_js_divergence,job_args)
    p.close()
    p.join() 
    return output




def Log2fold(x,y):
    x_mean = np.average(x)
    y_mean = np.average(y)
    diff_xy = np.log2(y_mean+1) - np.log2(x_mean+1)
    return diff_xy



def Plotting_Correlation(data_object,timepoint):
    real_data = data_object[data_object.obs["HPF"] == (timepoint)]
    pred_data =  data_object[data_object.obs["HPF"] == "predicted_" + str(timepoint)] 
    scanpy_obj_data = np.vstack((np.average(real_data.X,axis = 0),np.average(pred_data.X,axis = 0)))
    X = "Real Data at " + str(timepoint) + " HPF"
    Y = "Predicted Data at " + str(timepoint) +" HPF"
    scanpy_obj_data_df = pd.DataFrame(scanpy_obj_data.T,columns=[X,Y])
    corrfunc(scanpy_obj_data[0],scanpy_obj_data[1])
    sns.regplot(data = scanpy_obj_data_df,x = X ,y = Y,scatter=True)
    plt.show()


def Plotting_PCA(data_object):
	sc.tl.pca(data_object)
	sc.pp.neighbors(data_object, n_neighbors=25, n_pcs=50)
	sc.tl.louvain(data_object)
	sc.pl.pca(data_object, color=["hours post fertilization(HPF)"],size=200,alpha = 0.8)




def Predictions(training_object,test_object,Starting_point,End_point,method,VAE_obj):
	Training_data = training_object
	Test_data = test_object[test_object.obs['HPF'] == Starting_point]
	Real_data =  test_object[test_object.obs['HPF'] == End_point]
	if (method == "VAE+VA"):
		pred_data, delta_vector = VAE_obj.predict(adata= Training_data, adata_to_predict=Test_data,biased = False,condition_key = "HPF",
			conditions={"ctrl": Starting_point , "stim": End_point},obs_key="all",celltype_to_predict = None,cell_type_key = "HPF")

	elif(method == "LI"):
		pred_data, delta_vector = Linear_interpolations(adata= Training_data, adata_to_predict=Test_data,biased = False,condition_key = "HPF",
			conditions={"ctrl": Starting_point , "stim": End_point},obs_key="all",celltype_to_predict = None,cell_type_key = "HPF") 
		pred_data = np.array(pred_data)
		pred_data[pred_data < 0] = 0

	elif(method == "PCA+VA"):
		pred_data, delta_vector, pred_data_G = PCA_interpolations(adata= Training_data, adata_to_predict=Test_data,biased = False,condition_key = "HPF",
			conditions={"ctrl": Starting_point , "stim": End_point},obs_key="all",celltype_to_predict = None,cell_type_key = "HPF",pc_components = min((Test_data.shape) + (101,))-1 ) 
		pred_data = pred_data.T
		pred_data = np.array(pred_data)
		pred_data[pred_data < 0] = 0

	label = "predicted_" + str(End_point)
	pred_adata_obj = sc.AnnData(pred_data, obs={"HPF":[label]*len(pred_data)}, var={"var_names":Real_data.var_names})
	Test_data.obs['hours post fertilization(HPF)'] = str(Starting_point) + "HPF (Real Data)"   
	Real_data.obs['hours post fertilization(HPF)'] = str(End_point) + "HPF (Real Data)"
	pred_adata_obj.obs['hours post fertilization(HPF)'] = "Predicted Data at " + str(End_point) + "HPF" 
	Adata_obj = Real_data.concatenate(pred_adata_obj,Test_data)
	sc.pp.normalize_total(Adata_obj)
	return Adata_obj





def MSE(x, y, **kws):
    r = sklearn.metrics.mean_squared_error(x,y)    
    ax = plt.gca()
    ax.annotate("$MSE$ = {:.3f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
    ax.plot([-1, 1], [-1, 1],ls=':', c="orange")



def Log2Fold_change(data_object,startpoint,timepoint):
    starting_data = data_object[data_object.obs["HPF"] == startpoint]
    real_data = data_object[data_object.obs["HPF"] == timepoint]
    pred_data = data_object[data_object.obs["HPF"] == "predicted_" + str(timepoint)]
    p = Pool(processes=10)
    job_args = zip(np.array(starting_data.X.T),np.array(pred_data.X.T)) 
    output_1 = p.starmap(Log2fold,job_args)
    job_args = zip(np.array(starting_data.X.T),np.array(real_data.X.T)) 
    output_2 = p.starmap(Log2fold,job_args)
    p.close()
    p.join() 
    Log2fold_array = np.vstack((output_1,output_2))
    Y = "Predicted log fold change at " + str(timepoint) + "HPF"
    X = "Real log fold change at " + str(timepoint) + "HPF"
    scanpy_obj_data_df = pd.DataFrame(Log2fold_array.T,columns=[X,Y])
    scanpy_obj_data_df = scanpy_obj_data_df.replace([np.inf, -np.inf], np.nan)
    scanpy_obj_data_df.dropna(inplace=True)
    MSE(np.array(scanpy_obj_data_df[X]),np.array(scanpy_obj_data_df[Y]))
    #corrfunc(scanpy_obj_data[0],scanpy_obj_data[1])
    #sns.kdeplot(data = scanpy_obj_data_df,x = X ,y = Y)
    ax = sns.regplot(scanpy_obj_data_df[X],scanpy_obj_data_df[Y],marker='.',scatter_kws={'alpha':0.1})
    #plt.xlim(-2.5,2.5)
    #plt.ylim(-2.5,2.5)
    plt.show()



"""
def Box_Plot_JS(VAE,LI,PCA,timepoint):
    Methods = ["VAE + VA","LI","PCA+VA"]
    JS_VAE = multiprocessing_JS_function(VAE[VAE.obs["HPF"] == (timepoint)],VAE[VAE.obs["HPF"] == "predicted_" + str(timepoint)])
    JS_LI = multiprocessing_JS_function(LI[LI.obs["HPF"] == (timepoint)],LI[LI.obs["HPF"] == "predicted_" + str(timepoint)])
    JS_PCA = multiprocessing_JS_function(PCA[PCA.obs["HPF"] == (timepoint)],PCA[PCA.obs["HPF"] == "predicted_" + str(timepoint)])
    distance_array = np.vstack((JS_VAE,JS_LI,JS_PCA))
    JS_df = pd.DataFrame(np.array(distance_array).T,columns = Methods)
    plt.rcParams['figure.facecolor'] = 'white'
    b = sns.boxplot(data = JS_df,palette="coolwarm")
    #b.axes.set_title("4.7 HPF",fontsize=10)
    b.set_xlabel("Methods",fontsize=10)
    b.set_ylabel("JS distance",fontsize=10)
    b.tick_params(labelsize=8)
    plt.ylim(-0.1, 1)
    plt.show()
"""



def Box_Plot_JS(VAE,LI,PCA,timepoint):
    Methods = ["VAE + VA","LI","PCA+VA"]
    JS_VAE = multiprocessing_JS_function(VAE[VAE.obs["HPF"] == timepoint],VAE[VAE.obs["HPF"] == "predicted_" + str(timepoint)])
    JS_LI = multiprocessing_JS_function(LI[LI.obs["HPF"] == timepoint],LI[LI.obs["HPF"] == "predicted_" + str(timepoint)])
    JS_PCA = multiprocessing_JS_function(PCA[PCA.obs["HPF"] == timepoint],PCA[PCA.obs["HPF"] == "predicted_" + str(timepoint)])
    distance_array = np.vstack((JS_VAE,JS_LI,JS_PCA))
    JS_df = pd.DataFrame(np.array(distance_array).T,columns = Methods)
    plt.rcParams['figure.facecolor'] = 'white'
    b = sns.boxplot(data = JS_df,palette="coolwarm")
    #b.axes.set_title("4.7 HPF",fontsize=10)
    b.set_xlabel("Methods",fontsize=10)
    b.set_ylabel("JS distance",fontsize=10)
    b.tick_params(labelsize=8)
    plt.ylim(-0.1, 1)
    plt.show()
    


def Violin_Plot_JS(VAE,LI,PCA,timepoint):
    Methods = ["VAE + VA","LI","PCA+VA"]
    JS_VAE = multiprocessing_JS_function(VAE[VAE.obs["HPF"] == str(float(timepoint))],VAE[VAE.obs["HPF"] == "predicted_" + str(timepoint)])
    JS_LI = multiprocessing_JS_function(LI[LI.obs["HPF"] == str(float(timepoint))],LI[LI.obs["HPF"] == "predicted_" + str(timepoint)])
    JS_PCA = multiprocessing_JS_function(PCA[PCA.obs["HPF"] == str(float(timepoint))],PCA[PCA.obs["HPF"] == "predicted_" + str(timepoint)])
    distance_array = np.vstack((JS_VAE,JS_LI,JS_PCA))
    JS_df = pd.DataFrame(np.array(distance_array).T,columns = Methods)
    plt.rcParams['figure.facecolor'] = 'white'
    b = sns.violinplot(data = JS_df,palette="coolwarm")
    #b.axes.set_title("4.7 HPF",fontsize=10)
    b.set_xlabel("Methods",fontsize=10)
    b.set_ylabel("JS distance",fontsize=10)
    b.tick_params(labelsize=8)
    plt.ylim(-0.1, 1)
    plt.show()



def PCA_together(VAE_obj,LI_obj,PCA_obj,starting_timepoint,end_timepoint):
    PCA_X =  PCA_obj[PCA_obj.obs["HPF"] == "predicted_"+ str(end_timepoint)]
    PCA_X.obs["HPF"] = "PCA+VA"
    LI_X =  LI_obj[LI_obj.obs["HPF"] == "predicted_"+ str(end_timepoint)]
    LI_X.obs["HPF"] = "LI"
    VAE_X =  VAE_obj[VAE_obj.obs["HPF"] == "predicted_"+ str(end_timepoint)]
    VAE_X.obs["HPF"] = "VAE+VA"
    VAE_Y =  VAE_obj[VAE_obj.obs["HPF"] == str(float(end_timepoint))]
    VAE_Y.obs["HPF"] = str(end_timepoint)
    VAE_Z =  VAE_obj[VAE_obj.obs["HPF"] == str(starting_timepoint)]
    VAE_Z.obs["HPF"] = str(starting_timepoint)   
    Adata_all = VAE_X.concatenate(LI_X,PCA_X,VAE_Y,VAE_Z)
    sc.tl.pca(Adata_all)
    sc.pp.neighbors(Adata_all, n_neighbors=25, n_pcs=50)
    sc.tl.louvain(Adata_all)
    sc.pl.pca(Adata_all, color=["HPF"],size=200,alpha = 0.5)
   


