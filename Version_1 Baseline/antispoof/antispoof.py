from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
import cv2

def process(image):
  image = cv2.resize(image,(32,32))
  image = image.astype("float") / 255.0
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  return image

def antiSpoofingInit(MODEL_PATH):
  print("[INFO] loading liveness detector...")
  model = load_model(MODEL_PATH)
  return model


def antiSpoofing(image,model,face_det = True):
  # Face Detectoin
  text = image
  if face_det:
    faces = detector.detect_faces(image) # MTCNN
    if len(faces) == 0:
      print("[ERROR]: NO FACE DETECTED!")
      return image,text
    else:
      for face in faces:
        box = face["box"]
        # print(box)
        # topx,topy,height,width
        x1 = box[0] 
        y1 = box[1]
        x2 = x1+box[2]
        y2 = y1+box[3]
        confidence = face ["confidence"]
        # print(confidence)
        if confidence > 0.5: # SET Threshold
          roi = image[y1:y2,x1:x2]
  else:
    # ASSUME THAT FACE IS DIRECTLY PROVIDED
    roi = image
  
  if(roi is None):
    return image, text
  # cv2_imshow(roi)
  roi = process(roi)
  preds = model.predict(roi)[0]
  j = np.argmax(preds)
  if j == 1:
    text = "REAL"
    color = (0,255,0)
  else:
    text = "FAKE"
    color = (0,0,255)

  
  # cv2.putText(image, text, (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)
  # cv2.rectangle(image, (x1, y1), (x2, y2),color, 2)

  return text,color
