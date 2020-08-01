from imutils import paths
from PIL import  Image
import os 
import pickle
import torch
import cv2
import numpy as np

def cut_img(img):
	T_H = 64
	T_W = 64

	# A silhouette contains too little white pixels
	# might be not valid for identification.
	if img.sum() <= 10000:
		return None
	# Get the top and bottom point
	y = img.sum(axis=1)
	y_top = (y != 0).argmax(axis=0)
	y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
	# print(y_top,y_btm)
	img = img[y_top:y_btm + 1, :]
	# As the height of a person is larger than the width,
	# use the height to calculate resize ratio.
	_r = img.shape[1] / img.shape[0]
	_t_w = int(T_H * _r)
	img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
	# Get the median of x axis and regard it as the x center of the person.
	sum_point = img.sum()
	sum_column = img.sum(axis=0).cumsum()
	x_center = -1
	for i in range(sum_column.size):
		if sum_column[i] > sum_point / 2:
			x_center = i
			break
	if x_center < 0:
		return None
	h_T_W = int(T_W / 2)
	left = x_center - h_T_W
	right = x_center + h_T_W
	if left <= 0 or right >= img.shape[1]:
		left += h_T_W
		right += h_T_W
		_ = np.zeros((img.shape[0], h_T_W))
		# print(img.shape,_.shape)
		img = np.concatenate([_, img, _], axis=1)
	img = img[:, left:right]
	return img.astype('uint8')

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

def register(img,name,model):

	tta = False
	d = os.getcwd()
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
			emb = model(torch.tensor(img).view(1,1,64,64))
			emb_mirror = model(torch.tensor(mirror).view(1,1,64,64))
			embs.append(l2_norm(emb + emb_mirror))
		else:                        
			embs.append(model(torch.tensor(img).view(1,1,64,64)))

	embedding = torch.cat(embs).mean(0,keepdim=True)
	# embeddings.append(embedding)
	# names.append(name)
	# embeddings = torch.cat(embeddings)
	# names = np.array(names)
	print(embedding)
	emb_file_name = os.getcwd() + '\\gait\\gaitbank.pickle'
	# name_file_name = d + '\\faceRecognition\\names2.npy'

	if(os.path.exists(emb_file_name)):
		print("Loading Existing Embeddings...........")
		pickle_in = open(os.getcwd()+"\\gait\\gaitbank.pickle","rb")
		embed = pickle.load(pickle_in)
		embed[name] = embedding
		pickle_out = open(os.getcwd()+"\\gait\\gaitbank.pickle","wb")
		print("Serializing Embeddings.......")
		pickle.dump(embed, pickle_out)
		pickle_out.close()
	else:
		emb_all = {}
		emb_all[name] = embeddings
		pickle_out = open(os.getcwd()+"\\gait\\gaitbank.pickle","wb")
		print("Serializing Embeddings.......")
		pickle.dump(embed, pickle_out)
		pickle_out.close()

	print(name + "  Registered.........")
	return embeddings, names


def gait_infer(model,img,thresh,top=3):
	
	model.eval()
	pickle_in = open(os.getcwd()+"\\gait\\gaitbank.pickle","rb")
	embeddings = pickle.load(pickle_in)
	emb1 = model(torch.tensor(img).view(1,1,64,64)) 
	# print(emb1)
	pred = []
	# print(image.split('/')[-2])
	for key,emb2 in zip(embeddings.keys(),embeddings.values()):
		dist = (emb1 - emb2).pow(2).sum(1)
		# print(dist)
		if  dist < thresh:
			print(dist,"-->",key)
			pred.append((dist,key))
			thresh = dist
	if(top == 1):
		pred = sorted(pred,key=lambda x:x[0])[:1]
	if(top == 3):
		pred = sorted(pred,key=lambda x:x[0])[:3]
	
	# print(pred[1])
	return pred
   