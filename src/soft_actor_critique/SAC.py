"""
__author__ : Kumar shubham
__desc__   : loss and learning in soft actor critique
__date__   : 26-10-2019
written in tf2.0
"""
from agent import Policy,QValFn,ValFn
 


# import agent
import os 
import tensorflow as tf 
import tensorflow_probability as tfp
import copy

class SAC:
	def __init__(self,epoch,batchTr,batchVal,gamma,optimizer,modelName,logDir,lr,TAU):
		
		self.epoch = epoch
		self.gamma = gamma
		self.batchTr =  batchTr
		self.batchVal = batchVal
		self.opt = optimizer
		self.modelName = modelName
		self.logDir = logDir
		self.sumDir = os.path.join(self.logDir,self.modelName+"_summary")
		self.ckptDir = os.path.join(self.logDir,self.modelName+"_ckpt")
		self.lr = lr
		self.TAU = TAU

		OPTIMIZER= {
		"sgd": tf.keras.optimizers.SGD(learning_rate=self.lr,clipvalue=0.5),
		"Adam" : tf.keras.optimizers.Adam(learning_rate=self.lr,clipvalue=0.5),
		"rmsProp" : tf.keras.optimizers.RMSprop(learning_rate=self.lr,clipvalue=0.5),
		"adaGrad" : tf.keras.optimizers.Adagrad(learning_rate=self.lr,clipvalue=0.5)

		}
		LOSS = {
		"huber" : tf.keras.losses.Huber,
		"mse"   : tf.keras.losses.MSE,
		} 

		self.polOpt = OPTIMIZER[optimizer]
		self.qOpt   = OPTIMIZER[optimizer]
		self.vOpt   = OPTIMIZER[optimizer]


		####### network definition #############
		self.policy = Policy()
		self.QTarget = QValFn()
		self.QSample =QValFn()
		self.ValueFn = ValFn()


		############ summary Writer#############
		self.summary_writer = tf.compat.v2.summary.create_file_writer(os.path.join(self.sumDir,'logs/'), flush_millis=10000)
		self.summary_writer.set_as_default()
		self.global_step =  tf.compat.v1.train.get_or_create_global_step()

	def policyLoss(self,currentState):
		## define loss function for policy function
		# action = self.policy.samplePolicy(currentState,training=True)
		_,_,action,mean,varLog = self.policy.samplePolicy(currentState,training=True)
		# print ("policy Calc calculation :", mean["mean"]["enc_eps_5"])
		logPolicy = tf.stop_gradient(self.policy.lgOfPolicy(mean,varLog,action)) 
		# print(action["action"])
		qVal = self.QSample.QvalForward(currentState,action,training=False)
		policyLossOp = tf.reduce_sum(logPolicy-qVal)
		return policyLossOp
	def qValLoss(self,currentState,action,reward,nextState):

		#### NOTE function calculation depend on two key state and action  make sure they are consistent####
		# part of TODO ^^^
		## define loss function for Q value 
		#TODO : define data structure for current state next state and reward  
		#UPDATE : added few changes in Policy return to be consistent with value and Q function 

		vValNext = self.ValueFn.ValFnForward(nextState,training=False)
		qVal= self.QSample.QvalForward(currentState,action,training=True) ## stochastic sampling of state
		qTarget = reward + self.gamma*vValNext ## Question why not use QTarget instead of QVal
		
		loss = tf.reduce_sum(0.5*tf.pow((qVal-qTarget),2)) ## loss is explicitly defined for q based gradient  not for value function
		return loss 

	def vValLoss(self,currentState,action):
		## define loss function for value function

		value = self.ValueFn.ValFnForward(currentState,training=True)
		_,_,_,mean,varLog = self.policy.samplePolicy(currentState,training=False)
		qVal =  self.QSample.QvalForward(currentState,action,training=False)
		# print ("vavalue calculation :", mean["mean"]["enc_eps_5"]) 
		logPolicy = tf.stop_gradient(self.policy.lgOfPolicy(mean,varLog,action))
		softValue = tf.reduce_sum(qVal-logPolicy)
		
		return tf.reduce_sum(0.5*tf.pow((value-softValue),2))

	def flipQVal(self):
		## change the Q value from target to Sample
		#TODO : check if it is working or not 
		varName = [v.name for v in self.QSample.trainable_variables]
		for name in varName:
			self.QSample.trainable_variables[name] = self.target.trainable_variables[name]
		return 
		

	def softUpdate(self,locModel,tagModel):
		"""
		soft update the model parameters.
		theta_target = tau*theta_local + (1-tau)theta_target
		"""
		#TODO : check if its working or not 
		varName = [v.name for v in tagModel.trainable_variables]
		for name in varName:
			tagModel.trainable_variables[name] = self.TAU*tagModel.trainable_variables[name]+(1-self.TAU)*locModel.trainable_variables[name]
		return 





	def train(self,epState,batchState,batchReward,batchAction,batchNextState):
		# print(self.policy.finalModel.summary())
		# input()
		with tf.GradientTape(persistent=True) as tape:
			## training the model
			# TODO 
			##################### code for data queue ###################
			# batchState,batchReward,batchAction,batchNextState = 
			#############################################################

			lossPolicy = self.policyLoss(batchState)
			lossQValue = self.qValLoss(batchState,batchAction,batchReward,batchNextState)
			lossVvalue = self.vValLoss(batchState,batchAction)

			#TODO : modofy the policy model
			policyGradient = tape.gradient(lossPolicy,self.policy.finalModel.trainable_variables)
			QGradient = tape.gradient(lossQValue,self.QSample.finalModel.trainable_variables)
			ValGradient = tape.gradient(lossVvalue,self.ValueFn.finalModel.trainable_variables)

			
			# ValOldGrad = tf.keras.models.clone_model(self.ValueFn.finalModel)

			
			if (epState%3==0):
				############# ask GSR : better way of regularization ################
				print("policy")
				print (lossPolicy)
				# print("trainVar",self.policy.finalModel.trainable_variables)
				# print("gradient",policyGradient)
				self.polOpt.apply_gradients(zip(policyGradient, self.policy.finalModel.trainable_variables))
			elif(epState%3==1):
				print("qOpt")
				self.qOpt.apply_gradients(zip(QGradient, self.QSample.finalModel.trainable_variables))
			else:
				print("vVal")
				self.vOpt.apply_gradients(zip(ValGradient, self.ValueFn.finalModel.trainable_variables))
			# ## update the gradient of value function 
			# #TODO : check if its pointer based or you need to use deepcopy to initialize valOldGrad
			# self.softUpdate(locModel=valOldGrad, tagModel =self.ValueFn.finalModel)

			return lossPolicy,lossQValue,lossVvalue
