#Import Libraries

# antispoof,faceRecognition,gait,utility are local Libraries where functionality is implemented
from antispoof.antispoof import *
from faceRecognition.prepareFaceBank import *
from faceRecognition.faceRec import *
import cv2
import os
import torch
from utility import *
from faceRecognition.Pytorch_Retinaface.face_align import Detector
from mtcnn.mtcnn import MTCNN
from gait.model import *
from gait.inference import *


#Temperory function for now for gait recognition it must be removed after having proper segmentation model
def get_binary_gait(i):
  img = cv2.imread(os.getcwd()+"\\gait\\images\\"+str(i)+".PNG",cv2.IMREAD_UNCHANGED)
  return img


#Loading all the models in memory
#You Can find these functions in utility.py

anitspoof_model = load_antispoof_model()
face_model = load_face_model()
model_ft = load_gait_model()

#face detector
detector = MTCNN()

#retinaface for alignment
rf = Detector(os.getcwd()+"\\faceRecognition\\Pytorch_Retinaface\\weights\\Resnet50_Final.pth")

#Camera url 
#You can find this url in IP Webcam mobile app in IP Webcam --> more --> rstp url so that you dont have to use urllib library
url = 'rtsp://192.168.43.239:8080/h264_pcm.sdp'



#This is gait Threshold
thresh = 0.7
#Threshold for face module
threshold = 1.5
#Initial gate state
gate = "CLOSED"
# cap = cv2.VideoCapture(0)


#Utility Function for reading frame according to camera url 
def read_camera(url):
  #remove this if part when we have 3 distinct ip cameras i am using webcam for simplicity
  if(url == "0"):
    cap = cv2.VideoCapture(0)
    ret,img = cap.read()
    return img
  
  else:
    cap = cv2.VideoCapture(url)
    ret,img = cap.read()
    return img

#Start Of Main Script    
def main():
  j = 1
  i = 0
  cap = cv2.VideoCapture(0) 
  GEI = np.zeros((64,64))
  while True:
    try:
      flag = 0
      # Reading Cameras
      #Utility Function to read frames from multiple cameras it will return frame
      #read_camera() utility function

      ret,img = cap.read()


      #Showing Live feed without any visualization  
      cv2.imshow("live feed",img)
      cv2.waitKey(1)
      

      front_cam = img
      # side_cam = read_camera("0") # This will be replaced by read_camera(sideUrl) url for side cam when proper gait module is ready
      side_cam = front_cam # this line must be deleted after replacing above
      # head_cam = read_camera("0") # It is to be replaced with url of head camera Same as side cam scene replace with headURL
      head_cam = front_cam #Delete after replacing above line
      

      side_cam_copy  = side_cam.copy()
      side_cam_humans = 1 # For temperory case num numans detected by side cam is 1
      
      faces = detector.detect_faces(front_cam)
      heads = detector.detect_faces(head_cam)

      #If no face detected skip further processing capture next frame
      if(len(faces) == 0):
        print("Face Not Detected")
        continue


      #If no heads detected skip further processing capture next frame
      if(len(heads) == 0):
        print("Head Not Detected")
        continue

      # Calculating GEI for 10 frames

      if(j <= 10):
        # binary_image = get_binary_gait(side_cam) #When we have side cam
        binary_image = get_binary_gait(j) # For temperory once the segmentation model is done it can be replaced
        binary_image = cut_img(binary_image) # preprocessing over each binary image
        GEI += binary_image # adding in previous frames
        j+=1 # increamenting counter
        continue
      else:
        #Conveting to np array to be safe
        GEI = np.array(GEI)
        #Averagin the GEI by number of frames added
        GEI /= j
        #Taking Predictions from the gait model
        predictions = gait_infer(model_ft,GEI,thresh,top=3)
        #This is debug stuff
        print(predictions)
        #resetting counter to 1
        j = 1
        #Reinitializing the array with zeroes of shape(64,64)

        GEI = np.zeros((64,64))

      
      #After 10 frames proceed with recognition and verification
      # Only Considering Single Entry Point For Now
      if(not(len(heads)) == 1 or not(len(faces) == 1) or not(side_cam_humans == 1)):  
        gate = "CLOSE"
        img = draw_text_on_image(img,gate,20,40)
        cv2.imshow("live feed", img)
        continue

      else:
        # Considering detected face is one and so the head  
        for face in faces:
          # storing face detection coordinates for face
          x,y,orig_w,orig_h = face['box']
          
          # Cropping the face
          crop = front_cam[y:y+orig_h,x:x+orig_w].copy()
          face_copy = front_cam.copy()
          
          #Testing Face For Antispoofing
          text,color = antiSpoofing(crop,anitspoof_model,face_det=False)

          #Drawing everything on front image
          face_copy = draw_text_on_image(face_copy,text,x,y-10,color)
          face_copy = draw_rectangle_on_image(face_copy,x,y,orig_w,orig_h,color)
          
          #If FAKE then close the gate and process next frame
          # if(text == "FAKE"):
          #   gate = "CLOSE"
          #   face_copy = draw_text_on_image(face_copy,gate,20,40)
          #   cv2.imshow("POTENTIAL SPOOFING",face_copy)
          #   cv2.waitKey(1)
          #   #Update LOG file
          #   #Send Alert To the Security Personnel
          #   continue

          # If not Fake proceed with recognition 
          # Align Face  
          face_aligned = align_given_face(crop,rf)

          #If Something goes wrong with alignment   
          if(face_aligned is None):  
            print("Face was not found")
            continue
          
          # Storing height and width of aligned face
          h,w,_ = face_aligned.shape
          
          # Sanity check
          if(h<10 or w<10):
            continue

          # Preprocessing     
          # Resizing
          face_aligned = cv2.resize(face_aligned,(112,112))
          #Converting to PIL image format
          face_aligned = Image.fromarray(face_aligned)

          #Passing to model for face recognition
          results, score = infer(face_model, [face_aligned], threshold, True)
          
          #Name of identified person or unknown
          name = results
          
          #Drawing everything on image
          face_copy = draw_rectangle_on_image(face_copy,x,y,orig_w,orig_h,color)
          face_copy = draw_text_on_image(face_copy,name,x,y-70)

 

          #If Unknown
          if(name == 'Unknown'):
              print("UNKNOWN DETECTED")
              flag = 1

          if(flag != 1):
            if(name in ["Manish","Shah"]): #Instead Manish here comes predictions from gait module its just plcaeholder

              #Drawing Stuff On Sidecam Image
              side_cam_copy = draw_text_on_image(side_cam_copy,name,20,40)
              
              # Opening Gate
              gate = "OPEN"
              face_copy = draw_text_on_image(face_copy,gate,20,40)
              print("GATE :",gate)            
              
              area = abs(x+orig_w - y+orig_h) * abs(x -  y)
              #If bounding box areas becomes larger than specified value that means person is almost near to gate
              if(area>2800):
                print(" IN - ",name) 
                #pass # Update the log file]
              #Wait For Some time to allow person to enter and close the gate
              if i%20 == 0:
                gate = "CLOSE"
                face_copy = draw_text_on_image(face_copy,gate,20,40)

            else:
              # If the gait freatures of the person doesn't belong to the person recognized by face
              #then register him or give him temperory access
              register_gait_toggle = input("Do You Want To Register The Face(Y/N): ")
              
              #Register Process
              if(register_gait_toggle == "Y"):
                
                candidate_name = input("Enter The Name OF Candidate: ")
                _ , _ = register(GEI,candidate_name,model_ft)

              else:
                #If dont want to register then can give temperory access
                temp_access_toggle = input("Do You Want To Provide Temperory Access(Y/N): ")
                if(temp_access_toggle == "N"):
                  #If Temp access is also not allowed
                  #gate shall be closed
                  gate = "CLOSE"
                  print("GATE: ",gate)
                  face_copy = draw_text_on_image(face_copy,gate,20,40)  
                  side_cam_copy = draw_text_on_image(side_cam_copy,gate,20,40)  
                  continue
                  # pass # Don't open the gate and update log file
                else:
                  #If temp access granted
                  gate = "OPEN"
                  print("GATE :",gate)
                  face_copy = draw_text_on_image(face_copy,gate,20,40)  
                  side_cam_copy = draw_text_on_image(side_cam_copy,gate,20,40)  
                  #Update log file after 20 frames
                  if i%20 == 0:
                      pass # Update the log file
                  else:
                    gate = "CLOSE"
                    print("GATE: ",gate)
                if i%20 == 0:
                  gate = "CLOSE"
                  face_copy = draw_text_on_image(face_copy,gate,20,40)  
                  side_cam_copy = draw_text_on_image(side_cam_copy,gate,20,40)  
                  print("GATE :",gate)
                  pass # Update the log file


          else:
              #If Face is not recognized
              #Register Him or give him temp access
              register_toggle = input("Do You Want To Register The Face(Y/N): ")
              
              #if want to register
              if(register_toggle == "Y"):
                
                candidate_name = input("Enter The Name OF Candidate: ")
                #train the model
                _ , _ = register_face(face_aligned,candidate_name,face_model)

              else:
                
                #For providing temperory access
                temp_access_toggle = input("Do You Want To Provide Temperory Access(Y/N): ")
                if(temp_access_toggle == "N"):
                  #If dont want to provide temp access close the gate
                  gate = "CLOSE"
                  print("GATE: ",gate)
                  face_copy = draw_text_on_image(face_copy,gate,20,40)  
                  side_cam_copy = draw_text_on_image(side_cam_copy,gate,20,40)  
                  
                  # pass # Don't open the gate and update log file
                else:
                  #If temp access provided
                  gate = "OPEN"
                  print("GATE :",gate)
                  face_copy = draw_text_on_image(face_copy,gate,20,40)  
                  if i%20 == 0:
                      pass # Update the log file
                  else:
                    gate = "CLOSE"
                    print("GATE: ",gate)
                if i%20 == 0:
                  side_cam_copy = draw_text_on_image(side_cam_copy,gate,20,40)  
                  face_copy = draw_text_on_image(face_copy,gate,20,40)  
                  print("GATE :",gate)
                  pass # Update the log file

        #Showing the front cam output
        cv2.imshow("Cam1",face_copy)
        cv2.waitKey(1)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      i+=1
        
    except Exception as e:
      print(e)

#Executing Script
if __name__ == '__main__':
	  main()
