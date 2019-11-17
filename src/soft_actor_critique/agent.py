"""
__author__ : Kumar shubham
__date__   : 16 Oct 2019
__desc__   : code for the architecture of the model
## this is especialy for the problem of RL over Glow models

written in Tf2.0
"""

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

eps_name_size = {"enc_eps_0":(128, 128, 6),"enc_eps_1":(64, 64, 12),
							"enc_eps_2":( 32, 32, 24),
							"enc_eps_3":(16, 16, 48),
							"enc_eps_4":(8, 8, 96),
							"enc_eps_5": ( 4, 4, 384)}

ordLayProcs = ["enc_eps_0","enc_eps_1","enc_eps_2","enc_eps_3","enc_eps_4","enc_eps_5"]

"""
dictFormat = {
	"state"  :
	"action" :
	"next_state" :
	"done" : 
}
"""


#---------------defining the POlicy -------------------------
class Policy(object):
	def __init__(self,eps_layers=6,conv_size = 1):
		## total no of layers in a network
		self.eps_layers = eps_layers 
		self.finalModel = self.PolicyApprox()

	# @tf.function
	def fnApprox(self,layer_specific_mean=[],layer_specific_var=[],input_shape=[],batch_norm=True,dropout=False,drop_ratio=0.5 ):
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

			#meanApprox.add(tf.keras.layers.ReLU())

		meanModelOut = meanApprox(inputLayer)

		## creating the function for the variance calculation 
		varApprox = tf.keras.Sequential()

		for var_shape,var_kernel_size  in layer_specific_var:
			varApprox.add(tf.keras.layers.Conv2D(var_shape,var_kernel_size,padding="same",
								kernel_initializer=initializer))

			if batch_norm:
				varApprox.add(tf.keras.layers.BatchNormalization())

			if dropout:
				varApprox.add(tf.keras.Dropout(drop_ratio))

			#varApprox.add(tf.keras.layers.RelU())

		varModelOut = varApprox(inputLayer)
		## defining the output model of the mean vector
		meanModel = tf.keras.Model(inputs=inputLayer,outputs=meanModelOut)
		varModel  = tf.keras.Model(inputs=inputLayer,outputs=varModelOut)

		return (meanModel,varModel)
	# @tf.function
	def PolicyApprox(self):
		## function to approximate the policy for the network
		modelList = [self.fnApprox(layer_specific_mean=[(eps_name_size[name][2],1)],layer_specific_var=[(eps_name_size[name][2],1)],
							input_shape=eps_name_size[name])
							for name in ordLayProcs]

		## entropy calculation 
		# entropyList = [self.entropyCalc(inputShape=eps_name_size[name]) for name in ordLayProcs]
		
		# state list based prediction
		inputList= [tf.keras.layers.Input(shape=eps_name_size[name]) 
						for name in ordLayProcs]

		## list of the output (action for each individual element)
		meanOutList = []
		logVarOutList = []
		outputList = []
		actionList = []
		for model,inputState in zip(modelList,inputList):
			mean,var = model
			meanOut  = mean(inputState)
			varOut   = var(inputState)

			logVarOutList.append(varOut)
			meanOutList.append(meanOut)
			

			newValDist = tfp.distributions.Normal(loc =meanOut,scale=tf.pow(varOut,2))

			action = newValDist.sample()
			prob = newValDist.prob(action)
			actionList.append(action)			


		outputList  = [actionList,meanOutList,logVarOutList] ## appending the list of all output tensor
		finalModel  = tf.keras.Model(inputs=inputList,outputs=outputList)
		return finalModel
		

	def samplePolicy(self,inputDict,training):
		"""
		
		inputDict : dict of input state vector

		---return---
		return next state, and action resulting it

		"""
	# TOD:FULL CODE REVIEW
		action = {"action":{}}
		state  = {"states":{}}
		newState = {"states":{}}
		mean = {"mean":{}}
		varLog = {"varLog":{}}

		crtInput = [inputDict["states"][elm] for elm in ordLayProcs]

		actionOut,meanOut,varOut = self.finalModel(crtInput,training=training)
		for newVal,meanMat,varLogMat,layerValue,layerName in zip(actionOut,meanOut,varOut,crtInput,ordLayProcs):
			
			newState["states"][layerName] = layerValue+newVal
			action["action"][layerName] = newVal
			state["states"][layerName] = layerValue
			mean["mean"][layerName] = meanMat
			varLog["varLog"][layerName] = varLogMat

		
		return (newState,state,action,mean,varLog)

	def lgOfPolicy(self,meanDict,varLogDict,actionDict):
		## calculating the log of the policy for given parameter
		logVal = 0.0 
		for i,name in enumerate(ordLayProcs):
			mean = meanDict["mean"][name]
			varLog = varLogDict["varLog"][name]
			action = actionDict["action"][name]
			## TODO : DOUBT ABOUT THE CALCULATION OF THE VAL (tensorflow.python.framework.errors_impl.InvalidArgumentError:
			## Incompatible shapes: [1,128,128,6] vs. [1,64,64,12] [Op:AddV2] name: add/)
			val = tfp.distributions.Normal(loc=mean,scale = tf.pow(varLog,2),allow_nan_stats=False).prob(action)
			## TODO : Very bah hack of using the log here 
			#https://stackoverflow.com/questions/10343831/clamping-negative-logarithm-of-the-probability-to-a-positive-value-in-a-informat

			## TODO : there is some problem with 4X4X4 filter
			newVal = tf.ones_like(val)+val
			logVal+=tf.reduce_sum(tf.math.log(newVal))
			# print(newVal)
			###################################################BUG : fixed ########################################
			
			# print(logVal,newVal.shape)
			if tf.math.is_nan(logVal):
				print(newVal)
				print("mean : ",mean)
				print("varLog : ",tf.sqrt(tf.exp(varLog)) )
				input()
			### ask sir : about log cliping ####
		    #logValReturn =tf.math.maximum(logVal,tf.constant(-1e10, tf.float32))
			###############################################################################################
		
		print("logVal : ", tf.math.log(logVal))
		return tf.math.log(logVal)
	def policyLearn(self):
		## function for training the models 
		pass


#-----------------------------defining the Q-value --------------------------------------
class QValFn(object):
	## function for approximating the Q-value for a given state and action
	def __init__(self):

		self.finalModel = self.QvalueApprox()
	# @tf.function
	def fnApprox(self,layer_specific_param=[],input_shape=[],batch_norm=True,dropout=False,drop_ratio=0.5,denseLayer=512):
		##  function approximator, returning a concatenated output for given state and action processed file
		"""
		layer_specif_param : tuple of output dim and weight dim in the function approximator 
		input_shape : shape of the layers 
		
		-----Return-------
		model which take input state and action and return a dense layer output 

		"""


		state = tf.keras.layers.Input(shape=input_shape)
		action = tf.keras.layers.Input(shape=input_shape)

		initializer = tf.random_normal_initializer(0.,0.02)

		stateModel = tf.keras.Sequential()
		actionModel = tf.keras.Sequential()

		## model processing the state
		for output_shape,kernel_size  in layer_specific_param:
			stateModel.add(tf.keras.layers.Conv2D(output_shape, kernel_size, padding='same',
							 kernel_initializer=initializer))
			if batch_norm:
				stateModel.add(tf.keras.layers.BatchNormalization())
			if dropout:
				stateModel.add(tf.keras.Dropout(drop_ratio))
			stateModel.add(tf.keras.layers.ReLU())

		## doing flattening
		stateModel.add(tf.keras.layers.Flatten())
		stateModel.add(tf.keras.layers.Dense(denseLayer))

		stateModelOut = stateModel(state)

		## model processing the action
		for output_shape,kernel_size  in layer_specific_param:
			actionModel.add(tf.keras.layers.Conv2D(output_shape, kernel_size, padding='same',
							 kernel_initializer=initializer))
			if batch_norm:
				actionModel.add(tf.keras.layers.BatchNormalization())
			if dropout:
				actionModel.add(tf.keras.Dropout(drop_ratio))
			actionModel.add(tf.keras.layers.ReLU())
		
		## doing flattening
		actionModel.add(tf.keras.layers.Flatten())
		actionModel.add(tf.keras.layers.Dense(denseLayer))

		actionModelOut = actionModel(action)

		## feature concatenation 
		concat = tf.keras.layers.Concatenate()
		output = concat([stateModelOut,actionModelOut])

		fnApprox = tf.keras.Model(inputs=[state,action],outputs=output)
		return fnApprox
	# @tf.function
	def QvalueApprox(self,denseLayerInfo=[512,1]):
		## creating final Qvalue model which concatenate all the individual output and return a single QValue 

		"""
		denseLayerInfo : information about the dense layer

		--return--
		"""
		modelList = [self.fnApprox(layer_specific_param=[(eps_name_size[name][2],1)],input_shape= eps_name_size[name]) 
							for name in ordLayProcs]



		inputList= [(tf.keras.layers.Input(shape=eps_name_size[name]),tf.keras.layers.Input(shape=eps_name_size[name])) 
						for name in ordLayProcs]

		concatOut = []
		for model,outTensor in zip(modelList,inputList):
			state,action = outTensor
			output = model([state,action])
			concatOut.append(output)

		concatOut = tf.keras.layers.Concatenate()(concatOut)
		
		loopInput = concatOut
		for i,layerInfo in enumerate(denseLayerInfo):
			## extracting the features from the neural network

			if i !=len(denseLayerInfo)-1:
				loopInput = tf.keras.layers.Dense(layerInfo)(loopInput)
				loopInput = tf.keras.layers.ReLU()(loopInput)
			else:
				assert(layerInfo==1)
				loopInput = tf.keras.layers.Dense(layerInfo)(loopInput)

		output = loopInput
		finalModel  = tf.keras.Model(inputs=inputList,outputs=output)
		return finalModel

	def QvalForward(self,state,action,training):
		## approximating the value of Q value function.
		"""
		inpStateAct : dictionary with state, action 

		------result--------
		Q- value for a given state action pair
		"""
		modelInp = []
		for i,name in enumerate(ordLayProcs):
			modelInp.append(state["states"][name])
			modelInp.append(action["action"][name])

		return self.finalModel(modelInp,training=training)

	def QvalLearn(self):
		pass


# ---------------------------defining the value function --------------------------------
class ValFn(object):
	def __init__(self):
		self.finalModel = self.valFnApprox()
	# @tf.function
	def fnApprox(self,layer_specific_param=[],input_shape=[],batch_norm=True,dropout=False,drop_ratio=0.5,denseLayer=512):
		##  function approximator, returning a concatenated output for given state 
		"""
		layer_specif_param : tuple of output dim and weight dim in the function approximator 
		input_shape : shape of the layers 
		
		-----Return-------
		model which take input state and action and return a dense layer output 

		"""


		state = tf.keras.layers.Input(shape=input_shape)

		initializer = tf.random_normal_initializer(0.,0.02)

		stateModel = tf.keras.Sequential()
		## model processing the state
		for output_shape,kernel_size  in layer_specific_param:
			stateModel.add(tf.keras.layers.Conv2D(output_shape, kernel_size, padding='same',
							 kernel_initializer=initializer))
			if batch_norm:
				stateModel.add(tf.keras.layers.BatchNormalization())
			if dropout:
				stateModel.add(tf.keras.Dropout(drop_ratio))
			stateModel.add(tf.keras.layers.ReLU())

		## doing flattening
		stateModel.add(tf.keras.layers.Flatten())
		stateModel.add(tf.keras.layers.Dense(denseLayer))

		stateModelOut = stateModel(state)

		fnApprox = tf.keras.Model(inputs=state,outputs=stateModelOut)
		return fnApprox
	# @tf.function
	def valFnApprox(self,denseLayerInfo=[512,1]):
		## creating final Qvalue model which concatenate all the individual output and return a single QValue 

		"""
		denseLayerInfo : information about the dense layer

		--return--
		"""
		modelList = [self.fnApprox(layer_specific_param=[(eps_name_size[name][2],1)],input_shape= eps_name_size[name]) 
							for name in ordLayProcs]
		inputList= [(tf.keras.layers.Input(shape=eps_name_size[name])) 
						for name in ordLayProcs]

		concatOut = []
		for model,outTensor in zip(modelList,inputList):
			state = outTensor
			output = model([state])
			concatOut.append(output)

		concatOut = tf.keras.layers.Concatenate()(concatOut)
		
		loopInput = concatOut
		for i,layerInfo in enumerate(denseLayerInfo):
			## extracting the features from the neural network

			if i !=len(denseLayerInfo)-1:
				loopInput = tf.keras.layers.Dense(layerInfo)(loopInput)
				loopInput = tf.keras.layers.ReLU()(loopInput)
			else:
				assert(layerInfo==1)
				loopInput = tf.keras.layers.Dense(layerInfo)(loopInput)

		output = loopInput
		finalModel  = tf.keras.Model(inputs=inputList,outputs=output)
		return finalModel

	def ValFnForward(self,state,training):
		## approximating the value of value function.
		"""
		inpStateAct : dictionary with state, action 

		------result--------
		Q- value for a given state action pair
		"""
		modelInp = []
		for i,name in enumerate(ordLayProcs):
			modelInp.append(state["states"][name])

		return self.finalModel(modelInp,training=training)

	def ValFnLearn(self):
		pass
