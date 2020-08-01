import glob
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def trainTransform():
    return data_transforms['train']

def valTransform():
    return data_transforms['val']

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

def register_face(img,name,model, tta = True):
    
    d = os.getcwd()
    transformer = valTransform()
    model.eval()
    embeddings =  []
    embs = []
    
    with torch.no_grad():
        # for img in faces:
        img = np.array(img)
        h,w,_ = img.shape
        if(h<10 or w<10):
           print("Try Again ..........")
        else:   
            img = cv2.resize(img,(112,112))
            img = Image.fromarray(img)
            if tta:
                mirror = trans.functional.hflip(img)
                emb = model(transformer(img).to(device).unsqueeze(0))
                emb_mirror = model(transformer(mirror).to(device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(model(transformer(img).to(device).unsqueeze(0)))
    
    embedding = torch.cat(embs).mean(0,keepdim=True)
    embeddings.append(embedding)
    # names.append(name)
    embeddings = torch.cat(embeddings)
    # names = np.array(names)
    print(embeddings)
    emb_file_name = d + '\\faceRecognition\\facebank2.pth'
    name_file_name = d + '\\faceRecognition\\names2.npy'

    if(os.path.exists(emb_file_name)):
        print("Loading Existing Embeddings...........")
        emb = torch.load(emb_file_name)
        print(emb.size())
        emb = torch.cat((emb,embeddings),0)
        # print(emb.size())
        torch.save(emb,emb_file_name)
    else:
        torch.save(embeddings,emb_file_name)
	
    if(os.path.exists(name_file_name)):
        names = [name]
        names = np.array(names)
        names_file = np.load(name_file_name)
        names_file = list(names_file)
        names = np.concatenate((names_file,names),axis=0)
        np.save(name_file_name, names)
    else:
        names = ["Unknown",name]
        names = np.array(names)
        print(names)
        np.save(name_file_name,names)

    print(name + "  Registered.........")
    
    return embeddings, names
