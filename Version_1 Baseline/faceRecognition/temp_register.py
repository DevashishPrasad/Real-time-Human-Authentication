import glob

from faceRecognition.prepareFaceBank import *
from torchvision import transforms as trans
from PIL import Image
from faceRecognition.Pytorch_Retinaface.face_align import Detector
from torchvision import transforms
import numpy as np
import time
import torch
from torchsummary import summary
import cv2
from faceRecognition.Pytorch_ArcFace2.models import resnet101
import os   
from mtcnn.mtcnn import MTCNN

class HParams:
    def __init__(self):
        self.pretrained = False
        self.use_se = True

config = HParams()
face_model = resnet101(config)
face_model.load_state_dict(torch.load(d+"\\faceRecognition\\Pytorch_ArcFace2\\Model.pt",map_location='cpu'))
face_model.eval()

cap = cv2.VideoCapture(0)
detector = MTCNN()
while True:

	ret,img = cap.read()
	faces = detector.detect_faces(img)

	if(len(faces) == 0):
		# print("Face Not Detected")
		continue
	for face in faces:
		x,y,w,h = face['box']
		crop = img[y:y+h,x:x+w].copy()
		fac = rf.align(crop)
		if(len(fac)==0):
			print("Face was not found")
			continue
		
		fac = np.array(fac)
		face_aligned = fac[0]
		h,w,_ = face_aligned.shape
		if(h<10 or w<10):
			continue
		face_aligned = cv2.resize(face_aligned,(112,112))
		face_aligned = Image.fromarray(face_aligned)
		candidate_name = input("Enter The Name OF Candidate: ")
		prepare_facebank(face_aligned,candidate_name,face_model)
		break	
