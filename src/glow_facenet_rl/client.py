"""
@author : kumar shubham, IIIT-Bangalore
@date   : 10-10-2019
@desc   : making compatibility between tf 1.14 and tf 2.0 using flask 
"""
import requests
from PIL import Image
import numpy as np
import json
import time
def main():
	fileName1  = "/home/shubham/IIIT/glow/demo/test/img.png"
	imgObj1 = Image.open(fileName1)
	image = np.array(imgObj1)
	urlEncoder = "http://127.0.0.1:5000/encoder"
	urlDecoder  = "http://127.0.0.1:5000/decoder"
	resp = requests.post(urlEncoder,json = {"image":image.tolist()})
	outDict = json.loads(resp.content)

	for elm,val in outDict.items():
		print (elm,np.asarray(val).shape)
	## code for decoder
	start = time.time()
	requests.post(urlDecoder,json={"feature":outDict, "image":image.tolist()}) 
	end = time.time()
	print(end - start)
if __name__ =="__main__":
	main()