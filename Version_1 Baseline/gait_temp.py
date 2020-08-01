import cv2
import PIL
from PIL import Image
import torch
import numpy as np
from gait.inference import *
from gait.model import *
# url = ''


# import pixellib
# from pixellib.instance import instance_segmentation

# segment_image = instance_segmentation()
# segment_image.load_model(os.getcwd()+"\\gait\\mask_rcnn_coco.h5")

def get_binary_gait(i):
	img = cv2.imread(os.getcwd()+"\\gait\\images\\"+str(i)+".PNG",cv2.IMREAD_UNCHANGED)
	return img


chkpnt = torch.load(os.getcwd()+"\\gait\\sia_mnist_new.pth",map_location='cpu')
model_ft = SiaCNN()
model_ft.load_state_dict(chkpnt["model_state_dict"])
model_ft.eval()
thresh = 0.8
# acc = 0
i = 1;
# cap = cv2.VideoCapture(0)
while True:
	
	# try:
	# ret,img = cap.read()
	if(i <= 10):
		binary_image = get_binary_gait(i)
		binary_image = cut_img(binary_image)
		GEI += binary_image
		i+=1
		continue
	else:
		GEI = np.array(GEI)
		GEI /= i
		predictions = gait_infer(model_ft,GEI,thresh,top=3)
		print(predictions)
		i = 1
		GEI = np.zeros((64,64))

	# except Exception as e:
	# 	print(e)
