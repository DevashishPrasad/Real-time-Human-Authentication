from __future__ import unicode_literals

import cv2
import threading
from django.shortcuts import render
from django.http import JsonResponse
import base64
import os
# import timeit
import orjson as json
# from multiprocess import Pipe
# import time
# from background.views import hello
# from background_task import background
from SIH2020.models import Shared
#Creating child process using fork
# parent
import sys
import io
import numpy as np
# from kora.models import Image
# r, w = Pipe() 
# Create a pipe
# 

from PIL import Image
import cv2
# Take in base64 string and return cv image
def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

def visualize_head(img,coords):
  for coord in coords:
    x1,y1,x2,y2 = coord
    cv2.putText(img,'head',(int(x1),int(y1)-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2,cv2.LINE_AA)
    cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
  return img

def cv2base64(frame):
    ret, jpeg = cv2.imencode('.jpg', frame)
    buffer1 = base64.b64encode(jpeg).decode('ascii')
    return buffer1

colors = [(255,0,0),(0,0,255),(0,255,0),(255,0,255),(255,255,0)]

def index(request):
	# global count
	# hellso(repeat=1)
	T = Shared.objects.get(id=2)
	# print(type(T.dicts))
	# print(T.dicts)
	# tic = timeit.default_timer()
	foo = json.loads(T.dicts)

	img_face = stringToRGB(foo["Frames"][0])
	img_head = stringToRGB(foo["Frames"][1])

	dets = foo["Face_land"]

	# Draw eyes
	for en,det in enumerate(dets):
		cv2.circle(img_face,(det[0][0],det[0][1]),3,colors[(en+1)%5],2)
		cv2.circle(img_face,(det[1][0],det[1][1]),3,colors[(en+1)%5],2)

	# Draw names and spoofs
	for i,out in enumerate(zip(foo["spoof"],foo["Persons"])):	
		cv2.putText(img_face, str(out[0]), (dets[i][0][0],dets[i][0][1]+60), cv2.FONT_HERSHEY_SIMPLEX ,  
			0.6, colors[(i+1)%5], 2, cv2.LINE_AA)
		cv2.putText(img_face, str(out[1]), (dets[i][0][0],dets[i][0][1]+40), cv2.FONT_HERSHEY_SIMPLEX ,  
			0.6, colors[(i+1)%5], 2, cv2.LINE_AA)

	# Draw heads
	img_head = visualize_head(img_head,foo["Head_Box"])

	img_face = cv2base64(img_face)
	img_head = cv2base64(img_head)
	if request.is_ajax():
		return JsonResponse({"face":img_face,"head":img_head,"gait":img_face}, status = 200)
		

	return  render(request, 'index1.html')


#
# image = cv2.imread("/home/ayan_gadpal/Pictures/a.png")
# ret, jpeg = cv2.imencode('.jpg', image)
#         buffer1 = base64.b64encode(jpeg).decode('ascii')
# sudo fuser -k 8000/tcp
