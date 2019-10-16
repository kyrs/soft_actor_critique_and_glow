"""
__author__ : Kumar shubham
__date__   : 16 Oct 2019
__desc__   : code for the architecture of the model
## this is especialy for the problem of RL over Glow models

written in Tf2.0
"""

import tensorflow as tf

class Policy(object):
	def __init__(self,eps_layers=6,conv_size = 1):
		## total no of layers in a network
		self.eps_layers = eps_layers 
		self.eps_name_size = {"enc_eps_0":(128, 128, 6),"enc_eps_1":(64, 64, 12),
							"enc_eps_2":( 32, 32, 24),
							"enc_eps_3":(16, 16, 48),
							"enc_eps_4":(8, 8, 96),
							"enc_eps_5": ( 4, 4, 384)}

		## creating a list of model based on the input data
		self.modelDict = {name:self.fnApprox(layer_specific_mean=[(shape[2],1)],layer_specific_var=[(shape[2],1)],input_shape=shape) 
							for name,shape in self.eps_name_size}


	def fnApprox(self,layer_specific_mean=[],layer_specific_var=[],input_shape=[],batch_norm=True,dropout=False,drop_ratio=0.5 )
	"""
	layer_specific_mean : list of tuple with (output layer specifics and kernel shapes)
	layer_specific_var  : list of tuple with (output layer specifics and kernel shapes)
	input_shape : [rows,col,channel] specifics of the input matrix

	----Return----
	return the function approximator of the mean and variance  
	"""
	inputLayer = tf.keras.layers.Input(shape=input_shape)
	initializer = tf.random_normal_initializer(0.,0.02)

	meanApprox = tf.keras.Sequential()

	## creating the function for the mean approximation
	for output_shape,kernel_size  in layer_specific_mean:
		meanApprox.add(tf.keras.layers.Conv2D(output_shape, kernel_size, padding='same',
							 kernel_initializer=initializer))

		if batch_norm:
			meanApprox.add(tf.keras.layers.BatchNormalization())

		if dropout:
			meanApprox.add(tf.keras.Dropout(drop_ratio))

		meanApprox.add(tf.keras.layers.ReLU())

	meanModelOut = meanApprox(input_layer)

	## creating the function for the variance calculation 
	varApprox = tf.keras.Sequential()

	for var_shape,var_kernel_size  in layer_specific_var:
		varApprox.add(tf.keras.layers.Conv2D(var_shape,var_kernel_size,padding="same",
							kernel_initializer=initializer))

		if batch_norm:
			varApprox.add(tf.keras.layers.BatchNormalization())

		if dropout:
			varApprox.add(tf.keras.Dropout(drop_ratio))

		varApprox.add(tf.keras.layers.RelU())

	varModelOut = varApprox(inputLayer)
	## defining the output model of the mean vector
	meanModel = tf.keras.Model(inputs=inputLayer,outputs=meanModelOut)
	varModel  = tf.keras.Model(inputs=inputLayer,outputs=varModelOut)

	return (meanModel,varModel)



	def sampNexState(self,modelDict={},inputDict={}):
	"""
	modelDict : dict of model for finding the mean and variance of an action
	inputDict : dict of input state vector

	---return---
	return next state after taking an action
	"""
		outputDict = {}

		for layerName,layerValue in inputDict.items():
			meanModel,varModel = modelDict[layerName]

			meanMat = meanModel(value, Training=False)
			varLog  = varModel(value, Training=False)

			newVal = tf.random.normal(meanMat.shape,mean=meanMat,stdDev =tf.sqrt(tf.exp(varLog)))
			outputDict[layerName] = layerValue+newVal
		
		return outputDict

	def trainNet(self,loss):
		## function for training the models 
class QValFn(object):
	pass

class ValFn(object):
	pass
