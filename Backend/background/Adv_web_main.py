#!/usr/bin/python3
import os, sys, time
import cv2
from multiprocess import Pipe
from time import sleep
from torch import multiprocessing as mp
import argparse
from threading import Thread
import urllib.request
import cv2
import numpy as np
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = 'http://192.168.43.1:8080/shot.jpg'
url2 = 'http://192.168.43.68:8080/shot.jpg'

class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.frame=0
        self.stop=0
        self.url = src
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            imgResp = urllib.request.urlopen(self.url)
            imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
            self.frame = cv2.imdecode(imgNp, -1)
            if(self.frame is None):
                self.stop=1


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Threaded Face recognition for stream of images')
    parser.add_argument('--detector_model', default= 'retina_res', help='Setting the detector model [Default : retina mob]')
    parser.add_argument('--recognizer_model', default= 'FEarcface50', help='Setting the recognizer model [Default : arcface]')
    parser.add_argument('--antispoof_model', default= 'EfficientB3', help='Setting the antispoof model [Default : EfficientB3]')
    parser.add_argument('--head_model', default= 'YOLO', help='Setting the head count model [Default : yolov5]')
    parser.add_argument('--emb_path', default= './', help='Setting the path to save embedings [Default : "./"]')


    args = parser.parse_args()

    # file descriptors r, w for reading and writing
    r, w = Pipe() 
    r2, w2 = Pipe()

    processid = os.fork()
    # processid2 = os.fork()
    if processid:
        # This is the parent process 
        # Closes file descriptor w
        # if processid2:
        #     pass
        # else:
        #     from Bottleneck_GAIT import *
        #     video_Face = VideoStreamWidget("/media/devashish/Local Disk/Research/Projects/SIH Face recognition/test_vid.mp4")
        #     video_head = VideoStreamWidget("/media/devashish/Local Disk/Research/Projects/SIH Face recognition/test_vid.mp4")
        #     while video_Face.stop!=1 or video_head.stop!=1:
        #         # This is the child process
        #         print ("Child2 reading")
        #         img = r2.recv()
        #         print ("text2 =", img  )
                
        #         out = GAIT_main()
        #         print ("Child2 writing")
        #         w2.send(out)
        #         print ("Child2 closing")
        #     sys.exit(0)
        str1 = None
        while str1!='close':
            print ("Parent writing")
            # reg_id = input("Eneter the id to be regsisterd:")
            # if(reg_id != 'x'):
            #     w.send([reg_id])
            # else:
            w.send([])
            print ("Parent closing")
            # print ("Parent2 writing")
            # w2.send('abc')
            # print ("Parent2 closing")
            print ("Parent2 reading")
            str1 = r2.recv()
            print ("text2 =", str1  )
            # print ("Parent reading")
            # str1 = r.recv()
            # print ("text =", str1  )
        sys.exit(0)
    else:
        from Adv_FaceRec import *
        Rec = FaceRecog(args.detector_model, args.recognizer_model, args.antispoof_model, args.head_model, args.emb_path)
        video_Face = VideoStreamWidget(url)
        video_head = VideoStreamWidget(url2)
        time.sleep(2)
        while video_Face.stop!=1 or video_head.stop!=1:
            # This is the child process
            print ("Child2 reading")
            reg = r.recv()
            print ("text2 =", reg  )
            # if len(reg) > 0:
            #     # processid3 = os.fork()
            #     # if not processid3:
            #     print("Registering")
            #     x = threading.Thread(target = Rec.register, args = (reg, []))
            #     x.start()
            #     print("Registered")
            #     x.join()
            #         # sys.exit(0)
            out,img, img_head = Rec.recognize(video_Face.frame,video_head.frame)

            img = cv2.resize(img,(800,500))  
            img_head = cv2.resize(img_head,(800,500))
            # horizontal = np.hstack((img,img_head))
            horizontal = np.concatenate((img,img_head),axis=1)

            print ("Child2 writing")
            w2.send(out)
            print ("Child2 closing")
            cv2.imshow("alala",horizontal)
            cv2.waitKey(1)
            # cv2.imshow("lalal",img_head)
            # cv2.waitKey(1)
        print ("Child2 writing")
        w2.send('close')
        print ("Child2 closing")
        sys.exit(0)