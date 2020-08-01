import cv2
import os
import torch
from faceRecognition.Pytorch_Retinaface.face_align import Detector
from antispoof.antispoof import *
from faceRecognition.prepareFaceBank import *
from faceRecognition.faceRec import *

from gait.model import *
from gait.inference import *


def draw_text_on_image(img,text,x,y,color=(255,255,0)):
  cv2.putText(img,text,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
  return img 

def draw_rectangle_on_image(img,x,y,w,h,color = (255,255,0)):
  cv2.rectangle(img, (x,y), (x+w, y+h), color)
  return img

def align_given_face(img,rf):
  fac = rf.align(img)
  fac = np.array(fac)
  face_aligned = fac[0]
  return face_aligned

class HParams:
    def __init__(self):
        self.pretrained = False
        self.use_se = True

def load_face_model():
  config = HParams()
  face_model = resnet101(config)
  face_model.load_state_dict(torch.load(os.getcwd()+"\\faceRecognition\\Pytorch_ArcFace2\\Model.pt",map_location='cpu'))
  return face_model

def load_antispoof_model():
  ANTI_SPOOF_MODEL_PATH = os.getcwd() + "\\antispoof\\A.model"
  model = antiSpoofingInit(MODEL_PATH=ANTI_SPOOF_MODEL_PATH)   
  return model

def load_gait_model():
  chkpnt = torch.load(os.getcwd()+"\\gait\\sia_mnist_new.pth",map_location='cpu')
  model_ft = SiaCNN()
  model_ft.load_state_dict(chkpnt["model_state_dict"])
  model_ft.eval()
  return model_ft