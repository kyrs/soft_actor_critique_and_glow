"""
@author : kumar shubham, IIIT-Bangalore
@date   : 10-10-2019
@desc   : making compatibility between tf 1.14 and tf 2.0 using flask 
"""

## NOTE : FOR COMPUTATIONAL OPTIMIZATION WE ARE USING THE TARGET IMAGE IN MAIN FUNCTION

import requests
from PIL import Image
import numpy as np
import json
import time

urlEncoder = "http://127.0.0.1:5000/encoder"
urlDecoder = "http://127.0.0.1:5000/decoder"
urlReward = "http://127.0.0.1:5000/reward"
def send_numpy_network(input_dict,tensor=True):
	## convert numpy file to list
	return_dict = {}
	for elmKey,val in input_dict.items():
		if tensor:
			return_dict[elmKey] = val.numpy().tolist()
		else:
			 return_dict[elmKey] = val.tolist()
	return return_dict

def recieve_numpy_network(input_dict):
	## convert list to numpy array
	outDict = {}
	for elmKey,val in input_dict.items():
		npVal = np.array(val)
		outDict[elmKey] = npVal.astype(np.float32)
	return outDict


def main():

	fileName1  = "../../images/sh1.jpg"
	imgObj1 = Image.open(fileName1)
	testImg = imgObj1.resize((256,256)) 
	image = np.array(testImg)
	resp = requests.post(urlEncoder,json = {"image":image.tolist()})
	outDict = json.loads(resp.content)

	for elm,val in outDict.items():
		print (elm,np.asarray(val).shape)
	## code for decoder
	start = time.time()
	requests.post(urlDecoder,json={"feature":outDict, "image":image.tolist()}) 
	end = time.time()
	print(end - start)

def encoderVec(imagePath):
	## fetching encoder path 
	imgObj1 = Image.open(imagePath)
	testImg = imgObj1.resize((256,256))
	image = np.array(testImg)
	resp = requests.post(urlEncoder,json = {"image":image.tolist()})
	outDict = json.loads(resp.content)
	outDict=recieve_numpy_network(outDict)
	return outDict

def reward(outDict=[],genSaveFlag=0,step=0,pathDir="",LAMBDA=100000,DistUp=10000):
	## TODO : FIRST FLASK CALL TAKES SOME TIME 
	## TODO : EUCD DISTANCE IS NOT ZERO ( for some reason could be stochasticity in glow)
	## dictionary with the key value wuth corresponding keyname and value 

	# fileName1  = "../../images/sh1.jpg"
	# imgObj1 = Image.open(fileName1)
	# testImg = imgObj1.resize((256,256))
	# image = np.array(testImg)
	# resp = requests.post(urlEncoder,json = {"image":image.tolist()})
	# outDict = json.loads(resp.content)
	# outDict=recieve_numpy_network(outDict)

	start = time.time()
	reward = requests.post(urlReward,json={"feature":send_numpy_network(outDict),"genSaveFlag":genSaveFlag,"step":step,"pathDir":pathDir})
	end = time.time()
	totTime = end-start
	print(reward.content)
	outDict = json.loads(reward.content)
	

	##TODO : reward over matching faces functions
	# if len(outDict["facenet"][0])==1 and outDict["eucd"][0][0] <5000 :
	# 	return -100*np.log(outDict["eucd"][0][0]),outDict["eucd"][0][0],False
	# elif len(outDict["facenet"][0])==1 and outDict["eucd"][0][0] >5000 :
	# 	return -100*np.log(outDict["eucd"][0][0]),outDict["eucd"][0][0],False 

	# if outDict["eucd"][0][0]<100:
	# 	return LAMBDA*np.abs(outDict["facenet"][0][1])-outDict["eucd"][0][0]/10.0,outDict["eucd"][0][0],True
	# else:
	# 	return LAMBDA*np.abs(outDict["facenet"][0][1])-outDict["eucd"][0][0]/10.0,outDict["eucd"][0][0],False

	if len(outDict["facenet"][0])==1:
		return 0.0,0.0,0.0
	else:
		return 50,0.0,0.0


if __name__ =="__main__":
	reward()