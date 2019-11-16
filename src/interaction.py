"""
__author__ : Kumar shubham
__date__   : 11-11-2019
"""


import sys
import os 
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
"""
## TODO : 
 1. reward based code --done
 2. Interaction code -- done
 3. Code review 
 4. running the base code --done
 5. way of seeing the modification by the model -- gif
 6. saving the log file and saving the model
 7. optimal rewards value
 8. Doubt about the log normal exploding
 9. tensorbo
"""

sys.path.insert(0,"./soft_actor_critique")
from soft_actor_critique.SAC import SAC 
from soft_actor_critique.replay_buffer import ReplayBuffer
from glow_facenet_rl.client import reward,encoderVec

def main():
	obj = SAC(epoch=5,batchTr=200,batchVal=200,gamma=0.1,optimizer="adaGrad",modelName="abcd",logDir="../logs",lr=0.001,TAU=0.9)
	RplBuf = ReplayBuffer(maxlen=1000,seed=100)
	NOEPISODE = 1000
	EPISODELEN=50000
	
	######## Fill the replayBuffer ###############
	imgPath ="../images/sh1.jpg"

	dictofGenModel = encoderVec(imgPath)
	count = 10
	currentState = {"states":dictofGenModel}
	while(count):
		### TODO : CHANCE OF BUGS TO CREEP IN
		count= count-1
		output = obj.policy.samplePolicy(currentState,training=False)
		newState,state,action,mean,varLog = output
		
		# TODO : using the concept of avoiding disastorous action
		try:
			r,eud,done = reward(newState["states"])
		except Exception as e:
			print(e)
			r = -1e4
			done=False
			newState=currentState
		print(r),print(done)
		if r >-5000:
			RplBuf.add(currentState,action,r,newState,done)
			currentState=newState
			
		else:
			RplBuf.add(currentState,action,r,newState,done)
			bufOut = RplBuf.sample(1)
			currentState=bufOut[0].state
		
	### STARTING THE TRAINING ##

	eps = 0
	while (eps<NOEPISODE):
		eps+=1
		step =0
		while(step<EPISODELEN):
			step+=1
			### stochastic gradient descent
			bufOut = RplBuf.sample(1)
			currentState = bufOut[0].state 
			action = bufOut[0].action
			# print(action.keys())
			r = bufOut[0].reward
			newState = bufOut[0].next_state
			done = bufOut[0].done

			lossPolicy,lossQValue,lossVvalue = obj.train(epState=step,batchState=currentState,batchAction=action,batchReward=r,batchNextState=newState)
			print ("EPISODE NO : {:4d}  STEP NO : {:4d} POLICY LOSS : {:2f} QValLOSS : {:2f} VvalLOSS : {:2f}".format(eps,step,lossPolicy,lossQValue,lossVvalue))


if __name__ =="__main__":
	main()