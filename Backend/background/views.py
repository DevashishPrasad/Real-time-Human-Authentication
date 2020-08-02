from __future__ import unicode_literals
from django.shortcuts import render
import cv2
# Create your views here.
from django.shortcuts import render,HttpResponse
from multiprocess import Pipe
from time import sleep
from torch import multiprocessing as mp
from threading import Thread
from background_task import background
from SIH2020.models import Shared
import base64
import os
import time
import json 
from .Adv_FaceRec import *


i = 1
BASE = "/home/ayan_gadpal/Documents/TheEminents/vid_logs/"

# Create your views here.

# gait = cv2.VideoCapture("http://192.168.43.1:8080/video")
# print("Inside Nackgroound process")

FILE_OUTPUT = BASE + str(int(time.time()))+'face.avi'
FILE_OUTPUT2 = BASE + str(int(time.time()))+'head.avi'

def cv2base64(frame):
    ret, jpeg = cv2.imencode('.jpg', frame)
    buffer1 = base64.b64encode(jpeg).decode('ascii')
    return buffer1

class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.frame=0
        self.stop=0
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.status = False

    def update(self):
        # Read the next frame from the stream in a different thread
        for i in range(1000000):
            if(i%100 == 0):
                print(i)
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                # cv2.imshow("testjpg", self.frame)
                # cv2.waitKey(1)
            time.sleep(0.01)
        self.stop=1
    def release(self):
        self.capture.release()

video_Face = VideoStreamWidget("/home/ayan_gadpal/Documents/TheEminents/SIH2020/background/messi_interview.mp4")
video_head = VideoStreamWidget("/home/ayan_gadpal/Documents/TheEminents/SIH2020/background/messi_interview.mp4")

Rec = FaceRecog("retina_res", "FEarcface50", "EfficientB3", "YOLO", "./")

time.sleep(6)

@background()   
def hello():
    print ("Background started")
    while video_Face.stop!=1 or video_head.stop!=1:
        # This is the child process
        # reg = r.recv()
        # print ("text2 =", reg  )
        # if len(reg) > 0:
        #     # processid3 = os.fork()
        #     # if not processid3:
        #     print("Registering")
        #     x = threading.Thread(target = Rec.register, args = (reg, []))
        #     x.start()
        #     print("Registered")
        #     x.join()
                # sys.exit(0)
        print ("Main Dict Request sent")
        # cv2.imshow("testjpg", video_Face.frame)
        # cv2.waitKey(1)
        main_dict = Rec.recognize(video_Face.frame,video_head.frame)
        main_dict["Frames"][0] = cv2base64(main_dict["Frames"][0]) 
        main_dict["Frames"][1] = cv2base64(main_dict["Frames"][1]) 
        T = Shared.objects.get(id=2)
        temp_dict = json.dumps(main_dict)
        T.dicts = temp_dict
        T.save()
        print ("Main Dict Received")

def main(request):
    # print("Maine")
    # hello(repeat=10,repeat_until=None)
    return HttpResponse("Hello world !" + str(count))
