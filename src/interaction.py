"""
__author__ : Kumar shubham
__date__   : 11-11-2019
"""


import sys 
"""
## TODO : 
 1. reward based code 
 2. Interaction code
 3. Code review 
 4. running the base code
 5. way of seeing the modification by the model -- gif
 6. savibg the log file 
"""

sys.path.insert(0,"./soft_actor_critique")
from soft_actor_critique.SAC import SAC 
from soft_actor_critique.replay_buffer import ReplayBuffer
from glow_facenet_rl.client import main

def main():
	obj = SAC(epoch=5,batchTr=200,batchVal=200,gamma=0.9,optimizer="adaGrad",modelName="abcd",logDir="../logs",lr=0.001,TAU=0.9)


if __name__ =="__main__":
	main()