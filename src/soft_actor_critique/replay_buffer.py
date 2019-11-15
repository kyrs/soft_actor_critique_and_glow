"""
__author__ : Kumar shubham
__desc__   : code for storing the information about the interaction
__date__   : 26-10-2019
written in tf2.0
"""

from collections import deque,namedtuple
import random 

eps_name_size = {"enc_eps_0":(128, 128, 6),"enc_eps_1":(64, 64, 12),
							"enc_eps_2":( 32, 32, 24),
							"enc_eps_3":(16, 16, 48),
							"enc_eps_4":(8, 8, 96),
							"enc_eps_5": ( 4, 4, 384)}
ordLayProcs = ["enc_eps_0","enc_eps_1","enc_eps_2","enc_eps_3","enc_eps_4","enc_eps_5"]


class ReplayBuffer(object):
	def __init__(self,maxlen,seed):
		"""
		maxLen : max length of the deque
		seed   : defining the random state  
		"""

		self.maxlen = maxlen
		self.seed = random.seed(seed)
		# self.batchSize = batchSize
		self.memory = deque(maxlen = self.maxlen)
		self.experience = namedtuple("experience", ["state","action","reward","next_state","done"])

	def add(self,currentState,action,reward,nextState,done):
		## items to add into the deque 
		"""
		currentState : state where system is currently in 
		action 		 : action taken by the system in current state
		reward 		 : reward got by agent on taking given action
		nextState    : next state which agent moved to after taking given action 
		done 		 : flag pointing whether it has reached final state or not
		"""
		exp = self.experience(currentState,action,reward,nextState,done)
		self.memory.append(exp)


	def sample(self,batch):
		## method for sampling the data from the deque

		## OUTPUT :  vstacked sampled output
		batch = random.sample(self.memory,batch)

		# batchCurrentState= np.vstack([exp.currentState for exp in batch if exp is not None])
		# batchAction 	 = np.vstack([exp.action for exp in batch if exp is not None])
		# batchReward 	 = np.vstack([exp.reward for exp in batch if exp is not None])
		# batchNextState	 = np.vstack([exp.nextState for exp in batch if exp is not None])
		# batchDone 		 = np.vstack([exp.done for exp in batch if exp is not None])

		return batch

	def __len__(self):
		## defining the length option for the agent
		return len(self.memory)
