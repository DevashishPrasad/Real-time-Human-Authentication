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
d = os.getcwd()

class HParams:
    def __init__(self):
        self.pretrained = False
        self.use_se = True

config = HParams()
face_model = resnet101(config)
face_model.load_state_dict(torch.load(d+"\\faceRecognition\\Pytorch_ArcFace2\\Model.pt",map_location='cpu'))
face_model.eval()


rf = Detector(d+"\\faceRecognition\\Pytorch_Retinaface\\weights\\Resnet50_Final.pth")

detector = MTCNN()
cap = cv2.VideoCapture(0)

# while True:
# 	ret,img = cap.read()

# 	cv2.imshow("face",img)
# 	cv2.waitKey(1)
# 	faces = detector.detect_faces(img)

# 	if(len(faces) == 0):
# 		print("Face Not Detected")
# 		continue

# 	x,y,w,h = faces[0]['box']
# 	fd = img.copy()

# 	crop = img[y:y+h,x:x+w].copy()
# 	fac = rf.align(crop)

	
# 	if(len(fac)==0):
# 		print("Face was not found")
# 		continue

# 	fac = np.array(fac)
	
# 	face_aligned = fac[0]


# 	h,w,_ = face_aligned.shape

# 	if(h<10 or w<10):
# 		continue

# 	face_aligned = cv2.resize(face_aligned,(112,112))
	
# 	face_aligned = Image.fromarray(face_aligned)
	
# 	candidate_name = input("Enter The Name OF Candidate: ")
	
# 	_ , _ = prepare_facebank(face_aligned,candidate_name,face_model)
	
		
# 	break	
	
arr = torch.load(d + "\\faceRecognition\\facebank2.pth")
print(type(arr))