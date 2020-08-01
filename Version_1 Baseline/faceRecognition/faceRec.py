import cv2
from PIL import Image
import glob
import torch
# from matplotlib.pyplot import imshow
import numpy as np
from faceRecognition.Pytorch_Retinaface.face_align import Detector
from faceRecognition.Pytorch_ArcFace2.models import resnet101
from faceRecognition.prepareFaceBank import * 
import os

d = os.getcwd()
transformer = valTransform()

def infer(model, faces, threshold, tta=False):
    '''
    faces : list of PIL Image
    target_embs : [n, 512] computed embeddings of faces in facebank
    names : recorded names of faces in facebank
    tta : test time augmentation (hfilp, that's all)
    '''
    model.eval()
    target_file_name = d + "\\faceRecognition\\facebank2.pth"
    names_file_name = d + "\\faceRecognition\\names2.npy"

    if(not (os.path.exists(target_file_name))):
      print("Embedding File Not Present")
      return 

    if(not (os.path.exists(names_file_name))):
      print("Names File Not Present")


    targets = torch.load(d + "\\faceRecognition\\facebank2.pth")
    names = np.load(d + "\\faceRecognition\\names2.npy")
    embs = []

    if tta:
        mirror = trans.functional.hflip(faces[0])
        emb = model(transformer(faces[0]).to(device).unsqueeze(0))
        emb_mirror = model(transformer(mirror).to(device).unsqueeze(0))
        embs.append(l2_norm(emb + emb_mirror))
    else:                        
        embs.append(model(transformer(faces[0]).to(device).unsqueeze(0)))
    
    source_embs = torch.cat(embs)    

    # print("Source Embeddings: ")
    # print(source_embs)
    # print("Target Embeddings: ")
    # print(targets)
    
    diff = source_embs.unsqueeze(-1) - targets.transpose(1,0).unsqueeze(0)

    dist = torch.sum(torch.pow(diff, 2), dim=1)
    minimum, min_idx = torch.min(dist, dim=1)
    
    print("Minimum: ",minimum)
    print("Min idx: ",min_idx)
    
    min_idx[minimum > threshold] = -1 # if no match, set idx to -1
    print(min_idx)

    e_name = names[min_idx[0]+1]
    
    return e_name, minimum               
