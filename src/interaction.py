"""
__author__ : Kumar shubham
__date__   : 11-11-2019
"""


import sys
import os 
import tensorflow as tf
from datetime import datetime
import shutil 

os.environ["CUDA_VISIBLE_DEVICES"] = "1,0" 
"""
## TODO : 
 1. reward based code --done
 2. Interaction code -- done
 3. Code review 
 4. running the base code --done
 5. way of seeing the modification by the model -- gif
 6. saving the log file and saving the model --
 7. optimal rewards value 
 9. tensorboard -- DONE
 10: logging -- DONE
"""

sys.path.insert(0,"./soft_actor_critique")
from soft_actor_critique.SAC import SAC 
from soft_actor_critique.replay_buffer import ReplayBuffer
from glow_facenet_rl.client import reward,encoderVec
import pandas as pd

def fillReplayBuffer(count,rplObj,policyObj):
	## sample from the policy and fill up the replay buffer
	"""
	count     : no of data to fill in the buffer
	rplObj    : replay buffer object
	policyObj : policy object buffer
	currentState : current state to start with 
	"""
	currentState,_ = sampleCeleba()
	while(count):
		### TODO : CHANCE OF BUGS TO CREEP IN
		if ((count-1)%20 ==0):
			currentState,_ = sampleCeleba()
		else:
			pass
		count= count-1
		output = policyObj.policy.samplePolicy(currentState,training=False)
		newState,state,action,_,_,_,_ = output
		
		# TODO : using the concept of avoiding disastorous action
		
		print("here")
		try:
			r,eud,done = reward(outDict=newState["states"])  
			if r<-1e4:
				raise Exception("reward as infinity euclidean")
			expFlag=False
		except Exception as e:
			print(e)
			expFlag=True
			r = -1e4
			done=False
			newState=currentState
		
		print(r),print(done),print(count),print(expFlag)


		if r >-5000:
			rplObj.add(currentState,action,r,newState,done)
			currentState=newState
		else:
			rplObj.add(currentState,action,r,newState,done)
			bufOut = rplObj.sample(1)
			currentState=bufOut[0].state

		
	return 

def calcRewardMean(count,policyObj,currentState,pathDir="",genSaveFlag=False):
## sample from the policy and calculate the reward with the mean value.
	cumReward = 0
	step = 0
	while(count):
		### TODO : CHANCE OF BUGS TO CREEP IN
		step+=1
		count= count-1
		output = policyObj.policy.samplePolicy(currentState,training=False)
		_,_,_,_,_,_,actionState = output
		
		# TODO : using the concept of avoiding disastorous action
		
		print("here")
		try:
			r,eud,done = reward(outDict = actionState["rewardAction"],genSaveFlag=genSaveFlag,pathDir=pathDir,step=step)  
			if r<-1e4:
				raise Exception("reward as infinity euclidean")
			expFlag=False
		except Exception as e:
			print(e)
			
		currentState={}
		currentState["states"]=actionState["rewardAction"]
		
		cumReward+=r
		print ("CUMM REWARD :{:4f} ".format(cumReward))
		
	return cumReward

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
	return currentState,imgPath

def main():
	maxLen = 3000
	obj = SAC(gamma=0.99,optimizer="Adam",modelName="abcd",logDir="../logs",lr=0.004,TAU=0.5)
	RplBuf = ReplayBuffer(maxlen=maxLen,seed=50)
	NOEPISODE = 10000
	LOGIMG =50
	FILLBUFFER = 150
	TESTSTATE,TESTIMG  = sampleCeleba()
	LOGDIR = "/home/shubham/Desktop/shubham/thesis1/logs/abcd_summary/image"
	timeStr =datetime.now().strftime("%Y_%m_%d_%H_%M")
	saveImgPath = os.path.join(LOGDIR,timeStr)
	os.makedirs(saveImgPath)
	shutil.copy2(TESTIMG,saveImgPath)
	######## Fill the replayBuffer ###############
	fillReplayBuffer(count=FILLBUFFER,rplObj=RplBuf,policyObj=obj)
	#############################################	
	### STARTING THE TRAINING ##

	eps = 0
	step =1
	
	while (eps<NOEPISODE):
		eps+=1
		# EPISODELEN = min(len(RplBuf),maxLen)*3
		# print(len(RplBuf))	
		EPISODELEN=5000
		moduloCount = 0
		while(moduloCount<EPISODELEN):
			step+=1
			moduloCount+=1
			### stochastic gradient descent
			bufOut = RplBuf.sample(1)
			currentState = bufOut[0].state 
			action = bufOut[0].action
			# print(action.keys())
			r = bufOut[0].reward
			newState = bufOut[0].next_state
			done = bufOut[0].done
			 
			lossPolicy,lossQValue,lossVvalue = obj.train(epState=step,batchState=currentState,batchAction=action,batchReward=r,batchNextState=newState,DONE=done)
			print ("EPISODE NO : {:4d}  STEP NO : {:4d} STEP MODULO :{:2f} MODULO COUNT : {:2f}  POLICY LOSS : {:2f} QValLOSS : {:2f} VvalLOSS : {:2f}".format(eps,step,EPISODELEN,moduloCount,lossPolicy,lossQValue,lossVvalue))
			

			######## Fill the replayBuffer ###############
		
		fillReplayBuffer(count=FILLBUFFER,rplObj=RplBuf,policyObj=obj)

		if (eps+1)%LOGIMG==0:
			### saving generated images
			logImg = os.path.join(saveImgPath,str(eps))
			os.makedirs(logImg)
			cumReward = calcRewardMean(100,obj,TESTSTATE,logImg,1)
			obj.loggingReward(cumReward,eps)
		else:
			cumReward = calcRewardMean(100,obj,TESTSTATE)
			obj.loggingReward(cumReward,eps)

		#############################################	

if __name__ =="__main__":
	main()
	# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
