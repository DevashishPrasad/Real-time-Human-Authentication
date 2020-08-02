#!/usr/bin/python3
import os, sys, time
import cv2
from multiprocess import Pipe
from time import sleep
from torch import multiprocessing as mp
import argparse
from threading import Thread

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
        for i in range(10000):
            if(i%1000 == 0):
                print(i)
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                # cv2.imwrite("test.jpg", self.frame)
            time.sleep(0.1)
        self.stop=1
    def release(self):
        self.capture.release()


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
            #     w.send([])
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
        video_Face = VideoStreamWidget("messi_interview.mp4")
        video_head = VideoStreamWidget("messi_interview.mp4")
        time.sleep(2)
        while video_Face.stop!=1 or video_head.stop!=1:
            # This is the child process
            print ("Child2 reading")
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
            main_dict = Rec.recognize(video_Face.frame,video_head.frame)
            # print(main_dict)
            print ("Child2 writing")
            w2.send(main_dict)
            print ("Child2 closing")
            # cv2.imshow("alala",main_dict["Frames"][0])
            # cv2.waitKey(1)
        print ("Child2 writing")
        w2.send('close')
        print ("Child2 closing")
        sys.exit(0)