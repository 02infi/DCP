
lamda = np.shape(training_data.X)[1] / 100
coeff = [1,1,1,lamda]
def Train_all_models(coeff = [1,1,1,1]):
	models = ["MSE_KL","MSE_MMD","sMSE_KL","sMSE_MMD"] 
	sequence = np.array([[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1]])
	config = sequence * coeff
	for i in range(0,len(config)):
		zebra_obj = deep_predictor.DeepPredictor(training_data,latent_dim=100,hidden_layers=[800,800],workers=8,parameters=config[i])
		zebra_obj.train(epochs=100)
		x = map(str, sequence)    
		confi_name = ''.join(x)
		path = "/home/gaurav/Gaurav/Berlin/Deep_Learning/pytorchVAE/experiment/model/" + models[i] + "/Withtestdata/" + data + confi_name + ".pt"
		zebra_obj.save_model(path=path)





		

