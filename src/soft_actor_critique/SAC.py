"""
__author__ : Kumar shubham
__desc__   : loss and learning in soft actor critique
__date__   : 26-10-2019
written in tf2.0
"""
from agent import Policy,QValFn,ValFn
import os 
import tensorflow as tf 

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

	def policyLoss(self):
		## define loss function for policy function
		pass

	def qValLoss(self):
		## define loss function for Q value 
		pass

	def vValLoss(self):
		## define loss function for value function
		pass
