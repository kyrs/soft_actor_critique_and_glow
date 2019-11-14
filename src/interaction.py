"""
__author__ : Kumar shubham
__date__   : 11-11-2019
"""


import sys
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
"""
## TODO : 
 1. reward based code --done
 2. Interaction code
 3. Code review 
 4. running the base code
 5. way of seeing the modification by the model -- gif
 6. saving the log file and saving the model
"""

sys.path.insert(0,"./soft_actor_critique")
from soft_actor_critique.SAC import SAC 
from soft_actor_critique.replay_buffer import ReplayBuffer
from glow_facenet_rl.client import reward,encoderVec

def main():
	obj = SAC(epoch=5,batchTr=200,batchVal=200,gamma=0.9,optimizer="adaGrad",modelName="abcd",logDir="../logs",lr=0.001,TAU=0.9)
	RplBuf = ReplayBuffer(maxlen=1000,seed=100,batchSize=100)

	
	######## Fill the replayBuffer ###############
	imgPath ="../images/sh1.jpg"

	dictofGenModel = encoderVec(imgPath)
	count = 100
	currentState = {"states":dictofGenModel}
	while(count):
		count= count-1
		output = obj.policy.samplePolicy(currentState)
		newState,state,action,mean,varLog = output
		r,done = reward(newState["states"])
		replayBuffer.add(currentState,action,r,nextState,done)
		currentState=newState

if __name__ =="__main__":
	main()