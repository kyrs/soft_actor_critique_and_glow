"""
@author : kumar shubham, IIIT-Bangalore
@date   : 10-09-2019
@desc   : main file for doing all the manipulation in the genereatiove model and RL based environment
"""
import sys 
import os
from PIL import Image
import time
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from flask import Flask
from flask import request
import json
app = Flask(__name__)
### adding the respective module in to the memory 
sys.path.insert(0,"/home/shubham/IIIT/glow/demo")
sys.path.insert(0,"/home/shubham/IIIT/facenet/src")
############## importing respective files for each of the model ############
import rl_latent_manipulation as glowFetExtMod
import rl_based_feature_ext as faceNetMod


def send_numpy_network(input_dict):
	## convert numpy file to list
	return_dict = {}
	for elmKey,val in input_dict.items():
		return_dict[elmKey] = val.tolist() 
	return return_dict

def recieve_numpy_network(input_dict):
	## convert list to numpy array
	outDict = {}
	for elmKey,val in input_dict.items():
		npVal = np.array(val)
		outDict[elmKey] = npVal.astype(np.float32)
	return outDict

@app.route('/encoder',methods=['POST'])
def encoder():
	data = request.json
	# param = data["params"]
	arr = np.array(data["image"])
	# print (arr.shape)
	# print (arr)
	arr1 = arr.astype(np.uint8)
	imgObj1 = Image.fromarray(arr1)	
	start = time.time()
	img_glow_emb1 = glowFetExtMod.fetch_glow_embedding(imgObj=imgObj1)
	return json.dumps(send_numpy_network(img_glow_emb1))


@app.route('/decoder',methods=['POST'])
def decoder():
	data = request.json
	# param = data["params"]
	# arr = np.array(data["feature"])
	arr = np.array(data["image"])
	arr1 = arr.astype(np.uint8)
	imgObj1 = Image.fromarray(arr1)	
	
	img_glow_emb1 = recieve_numpy_network(data["feature"])
	img_glow_emb = glowFetExtMod.flatten_eps_dict(img_glow_emb1)
	### processing second image 
	t1 = time.time()
	tup = ("Smiling",0.65)
	new_glow_emb,image = glowFetExtMod.action_rl(emb=img_glow_emb,attr_tup=tup)	
	t2 = time.time()
	#### change code of decoder 
	dist = faceNetMod.face_compare(np.array(imgObj1),np.array(image))
	t3 = time.time()
	print(dist)
	# print("total time : %f,embedding time : %f, action time : %f, reward time : %f"%(time.time()-start,t1-start,t2-t1,t3-t2))
	return json.dumps({"dist":123})
if __name__ =="__main__":
	app.run()
