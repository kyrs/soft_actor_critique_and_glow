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
import tensorflow_addons as tfa
# tf.debugging.set_log_device_placement(True)

class SAC:
	def __init__(self,gamma,optimizer,modelName,logDir,lr,TAU,ALPHA=0.001):
		
		self.gamma = gamma
		self.opt = optimizer
		self.modelName = modelName
		self.logDir = logDir
		self.sumDir = os.path.join(self.logDir,self.modelName+"_summary")
		self.ckptDir = os.path.join(self.logDir,self.modelName+"_ckpt")
		self.lr = lr
		self.TAU = TAU
		self.ALPHA=ALPHA
		### implementation of polyak averaging
		# self.avg = tf.train.ExponentialMovingAverage(decay = self.TAU)
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

		self.polOpt = OPTIMIZER[self.opt]
		self.qOpt   = OPTIMIZER[self.opt]
		self.vOpt   = tfa.optimizers.MovingAverage(OPTIMIZER[self.opt],average_decay=self.TAU)
		# self.vOpt = OPTIMIZER[optimizer]

		####### network definition #############
		self.policy = Policy()
		self.QSample2 = QValFn()
		self.QSample1 =QValFn() #https://spinningup.openai.com/en/latest/algorithms/sac.html
		self.ValueFn = ValFn()


		############ summary Writer#############
		self.summary_writer = tf.compat.v2.summary.create_file_writer(os.path.join(self.sumDir,'logs/'), flush_millis=10000)
		self.summary_writer.set_as_default()
		self.global_step =  tf.compat.v1.train.get_or_create_global_step()

	def policyLoss(self,currentState):
		## CHECKED
		## define loss function for policy function
		## TODO : FORMULATION DOESN"T MATCH THE PAPER 
		_,_,action,mean,sqrtStd,gauss,rewardAction = self.policy.samplePolicy(currentState,training=True) ## TODO : CHECK THE ORDER FROM POLICY NETWORK
		logPolicy = tf.stop_gradient(self.policy.lgOfPolicy(mean,sqrtStd,gauss))  ## TODO : check implementation here also 
		qVal = self.QSample1.QvalForward(currentState,action,training=False)
		policyLossOp = tf.reduce_mean(tf.abs(self.ALPHA*logPolicy-qVal))
		return policyLossOp
	def qValLoss(self,Qnetwork,currentState,action,reward,nextState,DONE):

		## CHECKED 

		#### NOTE function calculation depend on two key state and action  make sure they are consistent####
		# part of TODO ^^^
		## define loss function for Q value 
		#TODO : define data structure for current state next state and reward  
		#UPDATE : added few changes in Policy return to be consistent with value and Q function 

		vValNext = self.ValueFn.ValFnForward(nextState,training=False)
		qVal = Qnetwork.QvalForward(currentState,action,training=True) ## stochastic sampling of state 
		qTarget = reward + self.gamma*(1-DONE)*vValNext ## Question why not use QTarget instead of QVal
		
		loss = tf.reduce_mean(tf.pow((qVal-qTarget),2)) ## loss is explicitly defined for q based gradient  not for value function
		return loss 

	def vValLoss(self,currentState):
		### Checked 
		## define loss function for value function
		#### https://spinningup.openai.com/en/latest/algorithms/sac.html
		value = self.ValueFn.ValFnForward(currentState,training=True)
		_,_,action,mean,sqrtStd,gauss,rewardAction = self.policy.samplePolicy(currentState,training=False) ## TODO : check the order 
		qVal1 =  self.QSample1.QvalForward(currentState,action,training=False)
		qVal2 = self.QSample2.QvalForward(currentState,action,training=False)
		qVal = tf.math.minimum(qVal1,qVal2)
		logPolicy = tf.stop_gradient(self.policy.lgOfPolicy(mean,sqrtStd,gauss))
		softValue = tf.reduce_sum(qVal-self.ALPHA*logPolicy)
		##TODO : POLYAK averaging
		return tf.reduce_mean(tf.pow((value-softValue),2))

	 
		

	def softUpdate(self,locModel,tagModel):
		"""
		soft update the model parameters.
		theta_target = tau*theta_local + (1-tau)theta_target
		"""
		#TODO : check if its working or not 
		
		for targetParam,localParam in zip(tagModel.trainable_variables,locModel.trainable_variables):
			print ("old :",targetParam)
			targetParam.assign(self.TAU*targetParam+(1-self.TAU)*localParam)
			print (print ("new :",targetParam)) 
			
		return


	def loggingQLoss(self,loss,step):
		tf.summary.experimental.set_step(step)
		tf.compat.v2.summary.scalar('qvalue_loss', tf.math.log(loss))
	def loggingVLoss(self,loss,step):
		tf.summary.experimental.set_step(step)
		tf.compat.v2.summary.scalar('Vvalue_loss', tf.math.log(loss))
	def loggingPLoss(self,loss,step):
		tf.summary.experimental.set_step(step)
		tf.compat.v2.summary.scalar('policy_loss', tf.math.log(loss))

	def loggingReward(self,reward,step):
		tf.summary.experimental.set_step(step)
		tf.compat.v2.summary.scalar('reward', reward)


	def train(self,epState,batchState,batchReward,batchAction,batchNextState,DONE):
		# print(self.policy.finalModel.summary())
		# input()
		
			## training the model
			## ttrick to fit model on smaller GPU

			
			if (epState%3==0):
				############# ask GSR : better way of regularization ################
				# with tf.device('/device:GPU:1'):
				with tf.GradientTape() as Ptape:
					lossPolicy = self.policyLoss(batchState)
					#TODO : modofy the policy model
					policyGradient = Ptape.gradient(lossPolicy,self.policy.finalModel.trainable_variables)
					self.polOpt.apply_gradients(zip(policyGradient, self.policy.finalModel.trainable_variables))
					self.loggingPLoss(lossPolicy,epState//3)
					return lossPolicy,0.0,0.0
			elif(epState%3==1):
				# with tf.device('/device:GPU:1'):
				with tf.GradientTape() as Qtape:
					countQNet=epState//2

					if countQNet%2==0:
						Qnetwork=self.QSample1
						strQ=1
					else:
						Qnetwork=self.QSample2
						strQ=2
					print("qOpt : ",strQ,countQNet)
					lossQValue = self.qValLoss(Qnetwork,batchState,batchAction,batchReward,batchNextState,DONE)
					QGradient = Qtape.gradient(lossQValue,Qnetwork.finalModel.trainable_variables)
					self.qOpt.apply_gradients(zip(QGradient, Qnetwork.finalModel.trainable_variables))
					self.loggingQLoss (lossQValue,epState//3)
					return 0.0,lossQValue,0.0
			else:
				print("vVal")
				# with tf.device('/device:CPU:0'):
				with tf.GradientTape() as ValueTape:
					lossVvalue = self.vValLoss(batchState)
					ValGradient = ValueTape.gradient(lossVvalue,self.ValueFn.finalModel.trainable_variables)
					self.loggingVLoss(lossVvalue,epState//3)
					self.vOpt.apply_gradients(zip(ValGradient, self.ValueFn.finalModel.trainable_variables))
					return 0.0,0.0,lossVvalue
						