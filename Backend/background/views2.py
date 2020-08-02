# from __future__ import unicode_literals
# from django.shortcuts import render
# import cv2
# # Create your views here.
# from django.shortcuts import render,HttpResponse
# from multiprocess import Pipe
# from time import sleep
# from torch import multiprocessing as mp
# from threading import Thread
# from background_task import background
# from SIH2020.models import Shared
# import base64
# import os
# import time
# import json 
# from .Adv_main import VideoStreamWidget

# import os, sys, time
# from multiprocess import Pipe
# from time import sleep

# i = 1
# BASE = "/home/ayan_gadpal/Documents/TheEminents/vid_logs/"

# # Create your views here.

# video_Face = VideoStreamWidget(0)
# video_head = VideoStreamWidget(2)
# # gait = cv2.VideoCapture("http://192.168.43.1:8080/video")
# # print("Inside Nackgroound process")

# FILE_OUTPUT = BASE + str(int(time.time()))+'face.avi'
# FILE_OUTPUT2 = BASE + str(int(time.time()))+'head.avi'

# def cv2base64(frame):
#     ret, jpeg = cv2.imencode('.jpg', frame)
#     buffer1 = base64.b64encode(jpeg).decode('ascii')
#     return buffer1


# r, w = Pipe() 
# r2, w2 = Pipe()

# processid = os.fork()


# @background(schedule=1)   
# def hello():
#     # global i
#     # global FILE_OUTPUT
#     # global FILE_OUTPUT2
#     # if os.path.isfile(FILE_OUTPUT):
#     #     os.remove(FILE_OUTPUT)
#     # if os.path.isfile(FILE_OUTPUT2):
#     #     os.remove(FILE_OUTPUT2)
#     # frameF = video_Face.frame
#     # frameH = video_head.frame
#     # frameG = video_head.frame

#     # if video_Face.status and video_head.status:
#     #     if i == 1:
#     #         widthF = frameF.shape[1]   # float
#     #         heightF = frameF.shape[0]

#     #         widthH = frameH.shape[1]   # float
#     #         heightH = frameH.shape[0]
#     #         global out
#     #         global out2
#     #         fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
#     #         out = cv2.VideoWriter(FILE_OUTPUT,fourcc, 20.0, (int(widthF),int(heightF)))
#     #         out2 = cv2.VideoWriter(FILE_OUTPUT2,fourcc, 20.0, (int(widthH),int(heightH)))
#     #         out.write(frameF)
#     #         out2.write(frameH)

#     #     else:
#     #         out.write(frameF)
#     #         out2.write(frameH)

#     #     i+=1
#     #     if i >= 10:
#     #         # "100,200,200,233}"
#     #         out.release()
#     #         out2.release()
#     #         print("saved video 1: ",FILE_OUTPUT)
#     #         print("saved video 2: ",FILE_OUTPUT2)
#     #         i = 1
#     #         FILE_OUTPUT = BASE + str(int(time.time()))+'face.avi'
#     #         FILE_OUTPUT2 = BASE + str(int(time.time()))+'head.avi'

#     #     T = Shared.objects.get(id=1)
#     #     faceF=cv2base64(frameF)
#     #     headF=cv2base64(frameH)
#     #     gait=cv2base64(frameG)
#     #     foo = {"face":faceF,"head":headF,"gait":gait}
#     #     T.dicts = json.dumps(foo)
#     #     # T.gait=cv2base64(frameG)
#     #     T.save()
#     #     # print("Saved!")
#     #     # print("Saved!")
#     #     # cv2.imshow("s",frameF)
#     #     # print(i)
#     #     if cv2.waitKey(1) & 0xFF == ord('q'):
#     #         head.release()
#     #         face.release()
#     #         # gait.release()
#     #         os._exit(0)
#     # else:
#     #     print("Camera Feed Not Found")
    
#     # processid2 = os.fork()
#     if processid:
#         # This is the parent process 
#         # Closes file descriptor w
#         # if processid2:
#         #     pass
#         # else:
#         #     from Bottleneck_GAIT import *
#         #     video_Face = VideoStreamWidget("/media/devashish/Local Disk/Research/Projects/SIH Face recognition/test_vid.mp4")
#         #     video_head = VideoStreamWidget("/media/devashish/Local Disk/Research/Projects/SIH Face recognition/test_vid.mp4")
#         #     while video_Face.stop!=1 or video_head.stop!=1:
#         #         # This is the child process
#         #         print ("Child2 reading")
#         #         img = r2.recv()
#         #         print ("text2 =", img  )
                
#         #         out = GAIT_main()
#         #         print ("Child2 writing")
#         #         w2.send(out)
#         #         print ("Child2 closing")
#         #     sys.exit(0)
#         str1 = None
#         while str1!='close':
#             print ("Parent writing")
#             # UI
#             # WILL UPDATE DATABASE WITH UNKNOWN ID AND FLAG
#             # 
#             reg_id = input("Eneter the id to be regsisterd:")
#             if(reg_id != 'x'):
#                 w.send([reg_id])
#             else:
#                 w.send([])
#             print ("Parent closing")
#             # print ("Parent2 writing")
#             # w2.send('abc')
#             # print ("Parent2 closing")
#             print ("Parent2 reading")
#             str1 = r2.recv()
#             print ("text2 =", str1  )
#             # print ("Parent reading")
#             # str1 = r.recv()
#             # print ("text =", str1  )
#         os._exit(0)
#     else:
#         from Adv_FaceRec import *
#         Rec = FaceRecog(args.detector_model, args.recognizer_model, args.antispoof_model, args.head_model, args.emb_path)

#         # time.sleep(2)
#         while video_Face.stop!=1 or video_head.stop!=1:
#             # This is the child process
#             print ("Child2 reading")
#             reg = r.recv()
#             print ("text2 =", reg  )
#             if len(reg) > 0:
#                 # processid3 = os.fork()
#                 # if not processid3:
#                 print("Registering")
#                 x = threading.Thread(target = Rec.register, args = (reg, []))
#                 x.start()
#                 print("Registered")
#                 x.join()
#                     # sys.exit(0)
#             out,img = Rec.recognize(video_Face.frame,video_head.frame)
#             print ("Child2 writing")
#             w2.send(out)
#             print ("Child2 closing")
#             cv2.imshow("alala",img)
#             cv2.waitKey(0)
#         print ("Child2 writing")
#         w2.send('close')
#         print ("Child2 closing")
#         os._exit(0)



# def main(request):
#     # print("Maine")
#     # hello(repeat=10,repeat_until=None)
#     return HttpResponse("Hello world !" + str(count))
