
from antispoof.antispoof import *
from faceRecognition.prepareFaceBank import *
from faceRecognition.faceRec import *
import cv2
import os
import torch
from tqdm import tqdm
from utility import *
from faceRecognition.Pytorch_Retinaface.face_align import Detector
from mtcnn.mtcnn import MTCNN

anitspoof_model = load_antispoof_model()
face_model = load_face_model()
detector = MTCNN()
rf = Detector(os.getcwd()+"\\faceRecognition\\Pytorch_Retinaface\\weights\\Resnet50_Final.pth")

url = 'rtsp://192.168.43.239:8080/h264_pcm.sdp'

threshold = 1.5
gate = "CLOSED"
cap = cv2.VideoCapture(0)

def main():

  i = 0
  while True:
    try:
      ret, img = cap.read()
      flag = 0
      
      faces = detector.detect_faces(img)

      if(len(faces) == 0):
        print("Face Not Detected")
        continue
      
      for face in faces:

        x,y,orig_w,orig_h = face['box']
        
        crop = img[y:y+orig_h,x:x+orig_w].copy()
        face_copy = img.copy()

        text,color = antiSpoofing(crop,anitspoof_model,face_det=False)

        
        face_copy = draw_text_on_image(face_copy,text,x,y-10,color)
        face_copy = draw_rectangle_on_image(face_copy,x,y,orig_w,orig_h,color)

        if(text == "FAKE"):
          cv2.imshow("POTENTIAL SPOOFING",face_copy)
          cv2.waitKey(1)
          continue

        face_aligned = align_given_face(crop,rf)
          
        if(face_aligned is None):  
          print("Face was not found")
          continue
        
        h,w,_ = face_aligned.shape
        
        if(h<10 or w<10):
          continue

        face_aligned = cv2.resize(face_aligned,(112,112))
        face_aligned = Image.fromarray(face_aligned)
        results, score = infer(face_model, [face_aligned], threshold, True)
        name = results
        
        face_copy = draw_rectangle_on_image(face_copy,x,y,orig_w,orig_h,color)
        face_copy = draw_text_on_image(face_copy,name,x,y-70)

        cv2.imshow("live feed",img)
        cv2.waitKey(1)
        
        if(name == 'Unknown'):
            print("UNKNOWN DETECTED")
            flag = 1

        if(flag != 1):
          gate = "OPEN"
          face_copy = draw_text_on_image(face_copy,gate,20,40)
          print("GATE :",gate)            
          
          area = abs(x+orig_w - y+orig_h) * abs(x -  y)
          if(area>2800):
            print(" IN - ",name) 
            #pass # Update the log file]
          if i%20 == 0:
            gate = "CLOSE"
            face_copy = draw_text_on_image(face_copy,gate,20,40)
        
        else:
            
            register_toggle = input("Do You Want To Register The Face(Y/N): ")
            
            if(register_toggle == "Y"):
              
              candidate_name = input("Enter The Name OF Candidate: ")
              _ , _ = prepare_facebank(face_aligned,candidate_name,face_model)

            else:
              
              temp_access_toggle = input("Do You Want To Provide Temperory Access(Y/N): ")
              if(temp_access_toggle == "N"):
              
                gate = "CLOSE"
                print("GATE: ",gate)
                face_copy = draw_text_on_image(face_copy,gate,20,40)  
                
                # pass # Don't open the gate and update log file
              else:
                
                gate = "OPEN"
                print("GATE :",gate)
                face_copy = draw_text_on_image(face_copy,gate,20,40)  
                if i%20 == 0:
                    pass # Update the log file
                else:
                  gate = "CLOSE"
                  print("GATE: ",gate)
              if i%20 == 0:
                face_copy = draw_text_on_image(face_copy,gate,20,40)  
                print("GATE :",gate)
                pass # Update the log file

      cv2.imshow("Cam1",face_copy)
      cv2.waitKey(1)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      i+=1
        
    except Exception as e:
      print(e)

if __name__ == '__main__':
	  main()
