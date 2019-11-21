"""
__author__ : Kumar shubham
__date__   : 11-11-2019
"""


import sys
import os 
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1,0" 
"""
## TODO : 
 1. reward based code --done
 2. Interaction code -- done
 3. Code review 
 4. running the base code --done
 5. way of seeing the modification by the model -- gif
 6. saving the log file and saving the model
 7. optimal rewards value
 8. Doubt about the log normal exploding : tensor mathematics : ask GSR 
 9. tensorboard -- DONE
 10: logging -- DONE
"""

sys.path.insert(0,"./soft_actor_critique")
from soft_actor_critique.SAC import SAC 
from soft_actor_critique.replay_buffer import ReplayBuffer
from glow_facenet_rl.client import reward,encoderVec
import pandas as pd

def fillReplayBuffer(count,rplObj,policyObj,currentState):
	## sample from the policy and fill up the replay buffer
	"""
	count     : no of data to fill in the buffer
	rplObj    : replay buffer object
	policyObj : policy object buffer
	currentState : current state to start with 
	"""
	epsReward=0
	while(count):
		### TODO : CHANCE OF BUGS TO CREEP IN
		count= count-1
		output = policyObj.policy.samplePolicy(currentState,training=False)
		newState,state,action,mean,varLog = output
		
		# TODO : using the concept of avoiding disastorous action
		try:
			r,eud,done = reward(newState["states"])
			if r<-1e4:
				raise Exception("reward as infinity euclidean")
		except Exception as e:
			print(e)
			r = -1e4
			done=False
			newState=currentState
		print(r),print(done),print(count)

		rplObj.add(currentState,action,r,newState,done)
		currentState=newState


		if r >-5000:
			rplObj.add(currentState,action,r,newState,done)
			currentState=newState
		else:
			rplObj.add(currentState,action,r,newState,done)
			bufOut = rplObj.sample(1)
			currentState=bufOut[0].state

		epsReward+=r
	return epsReward

def sampleCeleba(celebaCsvPath="/mnt/hdd1/shubham/thesis1/dataset/celeba/list_eval_partition.csv",idx=0,celebaImgDir="/mnt/hdd1/shubham/thesis1/dataset/celeba/img_align_celeba/img_align_celeba"):
	## csvDir : director with CSV data 
	## idx  : to choose train test or val data
	## directory where images are kept

	#TODO : euclidean distance read infinity in some case
	df = pd.read_csv(celebaCsvPath)	
	fileInfo = df.loc[df['partition']==idx]["image_id"].sample(1,replace=True).to_string()
	fileName = fileInfo.split(" ")[-1]
	imgPath = os.path.join(celebaImgDir,fileName)
	dictofGenModel = encoderVec(imgPath)
	currentState = {"states":dictofGenModel}
	return currentState

def main():
	maxLen = 1000
	obj = SAC(epoch=5,batchTr=200,batchVal=200,gamma=0.9,optimizer="Adam",modelName="abcd",logDir="../logs",lr=0.0001,TAU=0.9)
	RplBuf = ReplayBuffer(maxlen=maxLen,seed=100)
	NOEPISODE = 10000
	
	
	######## Fill the replayBuffer ###############
	currentState = sampleCeleba()
	cummEpsReward = fillReplayBuffer(count=50,rplObj=RplBuf,policyObj=obj,currentState=currentState)
	#############################################	
	### STARTING THE TRAINING ##

	eps = 0
	step =1
	
	while (eps<NOEPISODE):
		eps+=1
		EPISODELEN = min(len(RplBuf),maxLen)*10	
		# EPISODELEN=200
		while((step%EPISODELEN)!=0):
			step+=1
			### stochastic gradient descent
			bufOut = RplBuf.sample(1)
			currentState = bufOut[0].state 
			action = bufOut[0].action
			# print(action.keys())
			r = bufOut[0].reward
			newState = bufOut[0].next_state
			done = bufOut[0].done
			 
			lossPolicy,lossQValue,lossVvalue = obj.train(epState=step,batchState=currentState,batchAction=action,batchReward=r,batchNextState=newState,DONE=done)
			print ("EPISODE NO : {:4d}  STEP NO : {:4d} POLICY LOSS : {:2f} QValLOSS : {:2f} VvalLOSS : {:2f}".format(eps,step,lossPolicy,lossQValue,lossVvalue))
			

			######## Fill the replayBuffer ###############
		
		currentState = sampleCeleba()
		cummEpsReward= fillReplayBuffer(count=50,rplObj=RplBuf,policyObj=obj,currentState=currentState)
		obj.loggingReward(cummEpsReward,eps)
		#############################################	

if __name__ =="__main__":
	main()
	# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
