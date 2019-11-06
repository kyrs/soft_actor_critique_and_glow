"""
__author__ : Kumar shubham
__desc__   : loss and learning in soft actor critique
__date__   : 26-10-2019
written in tf2.0
"""
from agent import Policy,QValFn,ValFn
import os 
import tensorflow as tf 
import tensorflow_probability as tfp


class SAC:
	def __init__(self,epoch,batchTr,batchVal,gamma,optimizer,loss,modelName,logDir,lr):
		
		self.epoch = epoch
		self.gamma = gamma
		self.batchTr =  batchTr
		self.batchVal = batchVal
		self.opt = optimizer
		self.loss = loss
		self.modelName = modelName
		self.logDir = logDir
		self.sumDir = os.path.join(self.logDir,self.modelName+"_summary")
		self.ckptDir = os.path.join(self.logDir,self.modelName+"_ckpt")
		self.lr = lr
		
		OPTIMIZER= {
		"sgd": tf.keras.optimizers.SGD(learning_rate=self.lr),
		"Adam" : tf.keras.optimizers.Adam(learning_rate=self.lr),
		"rmsProp" : tf.keras.optimizers.RMSprop(learning_rate=self.lr),
		"adaGrad" : tf.keras.optimizers.Adagrad(learning_rate=self.lr)

		}
		LOSS = {
		"huber" : tf.keras.losses.Huber,
		"mse"   : tf.keras.losses.MSE,
		} 

		self.opt = OPTIMIZER[optimizer]
		self.loss = LOSS[loss]


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
		action = self.policy.samplePolicy(currentState,training=True)
		logPolicy = self.policy.lgOfPolicy(action,training=True) 
		qval = self.QSample.QvalForward(currentState,action,training=False)
		policyLossOp = tf.reduce_sum(logPolicy-qVal)
		return policyLossOp
	def qValLoss(self,currentState,action,reward,nextState):
		## define loss function for Q value 
		#TODO : define data structure for current state next state and reward
		vValNext = self.ValueFn.ValFnForward(nextState,training=False)
		qVal= self.QSample.QvalForward(currentState,action,training=True) ## stochastic sampling of state
		qTarget = reward + self.gamma*vValNext ## Question why not use QTarget instead of QVal
		
		loss = tf.reduce_sum(0.5*tf.pow(qVal-qTarget),2) ## loss is explicitly defined for q based gradient  not for value function
		return loss 

	def vValLoss(self,currentState,action):
		## define loss function for value function

		value = self.ValueFn.ValFnForward(currentState,training=True)
		policyParam = self.policy.samplePolicy(currentState,training=False)
		qVal =  self.QSample.QvalForward(currentState,action,training=False) 
		logPolicy = self.policy.lgOfPolicy(policyParam,action,training=False)
		softValue = tf.reduce_sum(qVal-logPolicy)
		
		return tf.reduce_sum(0.5*tf.pow((value-softValue),2))

	def flipQVal(self):
		#TODO : code to flip the QVal function
		pass
	def train(self):
		with tf.GradientTape() as gradTape:
			## training the model
			##################### code for data queue ###################
			batchState,batchReward,batchAction,batchNextState = 
			#############################################################

			lossPolicy = self.policyLoss(batchState)
			lossQValue = self.qValLoss(batchState,batchAction,batchReward,batchNextState)
			lossVvalue = self.vValLoss(batchState,batchAction)

			## TODO : define gradient tape for the policy 
			policyGradient = genTape.gradient(lossPolicy,self.policy.)
			QGradient = genTape.gradient(lossQValue,self.QSample.finalModel.trainable_variables)
			ValGradient = genTape.gradient(lossVvalue,self.ValueFn.finalModel.trainable_variables)

			## TODO : define intial condition for valOldGrad
			ValOldGrad = ValGradient

			##TODO : How to do this part ??
			NewValGradient = 
			## TODO : define optimizer 
			self.polOpt.apply_gradients(zip(policyGradient, self.policy.finalModel.trainable_variables))
			self.qOpt.apply_gradients(zip(QGradient, self.QSample.finalModel.trainable_variables))
			self.vOpt.apply_gradients(zip(NewvalGradient, self.ValueFn.finalModel.trainable_variables))
			## update the gradient of value function 


			ValOldGrad = ValGradient