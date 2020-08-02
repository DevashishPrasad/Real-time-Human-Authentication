#from antispoof.antispoof import *
import cv2
import os
import queue
from .utility import *
from mtcnn.mtcnn import MTCNN
from .FaceLandmarking import Landmarkers
from .FaceRecognition import Recognizers
from .FaceUtils import Utils
from PIL import Image
import threading
import argparse
from .AntiSpoofing.AntiSpoofing import *
from .headlib.head_det import HeadDetection 
from .headlib.visualize import visualize 
import timeit

def sort_func(e):
    return e[0][0]

def pre(img, det, recognizer, antispoof_model, i):
    que = queue.Queue()
    face = Utils.align_face(img, det, pad=0.34)
    # cv2.imwrite('img.jpg',face)
    #det = recognizer.preprocess(face)
    x = threading.Thread(target = lambda q, arg : q.put(recognizer.preprocess(arg)), args = (que,face))
    x.start()
    face2 = antispoof_model.preprocess(face)
    x.join()
    det = que.get()
    return face2, det, i

def head_thread(head_model,img_head):
    img_head2 = head_model.preprocess(img_head)
    if(head_model.modelArch == 'efficientdet'):
        count, bboxes, scores = head_model.predict(img_head2[0],img_head2[0],img_head2[0],img_head2[1],img_head2[2])
    else:
        count, bboxes, scores = head_model.predict(img_head2,img_head,img_head2,img_head2,img_head2)
    return count, bboxes

class FaceRecog:

    que = queue.Queue()


    def __init__(self, landmarker_model, recognizer_model, antispoof_model, head_model, emb_path):
        self._landmarker = Landmarkers.get_landmarker(landmarker_model)
        self._recognizer = Recognizers.get_recognizer(recognizer_model, emb_path, create_new = False)
        self._antispoof_model = AntiSpoofing()
        self._antispoof_model.setModel( antispoof_model, Gpu=0.05)
        self._antispoof_model = self._antispoof_model.model
        #Create object of class HeadDetection
        self._head_model = HeadDetection()
	    #Load Model From Available Choices
        self._head_model.load_model(head_model) #efficientDet')

    def recognize(self, img, img_head):
        img = cv2.resize(img,(img.shape[1]//3,img.shape[0]//3))
        img_head = cv2.resize(img_head,(img_head.shape[1]//3,img_head.shape[0]//3))
        main_dict = {"Frames":[],"Head_Box":[],"Face_land":[],"spoof":[],"Persons":[],"unk_ids":[]}          
        torch.cuda.empty_cache()
        x = threading.Thread(target = lambda q, arg : q.put(head_thread(arg[0], arg[1])), args = (self.que,[self._head_model,img_head]))
        x.start()
        dets = self._landmarker.detect_landmarks(img)
        x.join()
        count, bboxes = self.que.get()
        # img_head = visualize(img_head, bboxes)
        # print(count)
        main_dict["Head_Box"] = bboxes
        count = len(dets) # for now
        if count == len(dets) and count>0:
            threads = []
            faces = []
            mirrors = []
            pred = []
            out = []
            dets = sorted(dets, key=sort_func)
            faces = [None] * len(dets)
            mirrors = [None] * len(dets)
            pred = [None] * len(dets)
            colors = [(255,0,0),(0,0,255),(0,255,0),(255,0,255),(255,255,0)]
            
            for en,det in enumerate(dets):
                # print((det[0][0],det[0][1]))

                main_dict["Face_land"].append([
                    (int(det[0][0]),int(det[0][1])),(int(det[1][0]),int(det[1][1])),(int(det[2][0]),int(det[2][1])),(int(det[3][0]),int(det[3][1])),(int(det[4][0]),int(det[4][1]))])
                
            for i in range(len(dets)):
                x = threading.Thread(target = lambda q, arg : q.put(pre(arg[0], arg[1], arg[2], arg[3], arg[4])), args = (self.que, [img, dets[i], self._recognizer, self._antispoof_model, i]))
                x.start()
                threads.append(x)
            for t in threads:
                t.join()
            for i in range(len(dets)):
                face, det, j = self.que.get()
                #cv2.imshow("result",face)
                #cv2.waitKey(1)
                # print("JJ ::",j)
                face_img, mirror = det
                # cv2.imwrite('img'+str(i)+'.jpg',np.array(img.permute(1, 2, 0).cpu().detach())*255)
                # cv2.imwrite('mirror'+str(i)+'.jpg',mirror)
                faces[j] = face_img
                mirrors[j] = mirror #self._recognizer.preprocess(face))
                pred[j] = face
                #text = 'REAL'
                #color = 0
                #print(text, color)
                #if text == 'REAL':
            faces = torch.stack(faces)
            mirrors = torch.stack(mirrors)
            x = threading.Thread(target = lambda q, arg : q.put(self._recognizer.recognize_faces_parallel(arg[0], arg[1], arg[2])), args = (self.que,[faces, mirrors, 1.25]))
            x.start()
            pred = torch.stack(pred)
            out.append(self._antispoof_model.process(pred)) 
            x.join()   
            out.append(self.que.get())      # self._recognizer.recognize_faces_parallel(faces,1.5))
            j = 0
            # Visualization            
            for i in range(len(dets)):
                main_dict["Persons"].append(int(out[1][0][i].item()))
                if(out[1][0][i] == -1):

                    main_dict["unk_ids"].append(int(out[1][1][j]))
                    j+=1
                
                main_dict["spoof"].append(int(out[0][i].item()))
                # cv2.putText(img, str(out[1][0][i]), (dets[i][0][0],dets[i][0][1]+40), cv2.FONT_HERSHEY_SIMPLEX ,  
                #     0.7, colors[(i+1)%5], 2, cv2.LINE_AA)
                # cv2.putText(img, str(out[0][i]), (dets[i][0][0],dets[i][0][1]+60), cv2.FONT_HERSHEY_SIMPLEX ,  
                #     0.7, colors[(i+1)%5], 2, cv2.LINE_AA)

            # cv2.imwrite("lala.jpg",img)
            main_dict["Frames"].append(img)
            main_dict["Frames"].append(img_head)
            return main_dict
        return main_dict

    def register(self, unkn_id, del_id):
        self._recognizer.register_unknowns(unkn_id, del_id)

def Face_recog_main(img_path, img_head, landmarker_model, recognizer_model, antispoof_model, head_model, emb_path):
    Recog = FaceRecog(landmarker_model, recognizer_model, antispoof_model, head_model, emb_path)
    while True:
        img = cv2.imread(img_path)
        img_head = cv2.imread(img_head)
        tic = timeit.default_timer()
        out = Recog.recognize(img, img_head)
        # Recog.register([],[1])
        # Recog.register(img_head,2)
        toc = timeit.default_timer()
        print(out)
        print("[TOTAL TIME] : ",toc-tic)
        # break
        return out


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Threaded Face recognition for stream of images')
    parser.add_argument('image_path', help='Input image for face recognition model')
    parser.add_argument('image_head', help='Input image for face recognition model')
    parser.add_argument('--landmarker_model', default= 'retina_res', help='Setting the landmarker model [Default : retina mob]')
    parser.add_argument('--recognizer_model', default= 'FEarcface50', help='Setting the recognizer model [Default : FEarcface50]')
    parser.add_argument('--antispoof_model', default= 'EfficientB3', help='Setting the antispoof model [Default : EfficientB3]')
    parser.add_argument('--head_model', default= 'YOLO', help='Setting the head count model [Default : yolov5]')
    parser.add_argument('--emb_path', default= './', help='Setting the path to save embedings [Default : "./"]')


    args = parser.parse_args()
    Face_recog_main(args.image_path, args.image_head, args.landmarker_model, args.recognizer_model, args.antispoof_model, args.head_model, args.emb_path)