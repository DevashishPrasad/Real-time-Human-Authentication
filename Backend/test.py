
import cv2

camera = cv2.VideoCapture("tommy.mp4")

def cv2base64(frame):
    ret, jpeg = cv2.imencode('.jpg', frame)
    buffer1 = base64.b64encode(jpeg).decode('ascii')
    return buffer1

# cv2.imshow("Tommy",cv2.imread("a.png"))
# http://192.168.43.1:8080

while 1:
    success, frame = camera.read()
    print(success)
    print("Why stop?")
    if success:
        print("if?")
        cv2.imshow("Tommy",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera.release()
            cv2.destroyAllWindows()
            break
        # cv2.waitKey(1)
    # else:
    #     cv2.destroyAllWindows()
    #     break

    #     # cv2.waitKey(0)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # else:
    #     print("else?")

    #     break


# import timeit
# import ast
# import json 
# li = {
#     "_id": "5f234b64ca2b1ab718cca3ef",
#     "index": 0,
#     "guid": "bfb3c557-a9ed-43f0-9027-1cf22ee58d34",
#     "isActive": True,
#     "balance": "$3,912.53",
#     "picture": "http://placehold.it/32x32",
#     "age": 20,
#     "eyeColor": "brown",
#     "name": "Tiffany Gonzalez",
#     "gender": "female",
#     "company": "ZYTREX",
#     "email": "tiffanygonzalez@zytrex.com",
#     "phone": "+1 (802) 544-3062",
#     "address": "271 Neptune Court, Dunbar, Texas, 4538",
#     "about": "Elit cupidatat sunt esse anim do ut consequat quis nisi laboris. Quis voluptate nisi officia incididunt elit amet esse. Sunt aliqua aliqua laboris ipsum minim nulla esse irure non.\r\n",
#     "registered": "2018-11-18T04:42:23 -06:-30",
#     "latitude": 34.40223,
#     "longitude": -138.252852,
#     "tags": [
#       "aliqua",
#       "officia",
#       "deserunt",
#       "sit",
#       "elit",
#       "est",
#       "officia"
#     ],
#     "friends": [
#       {
#         "id": 0,
#         "name": "Britt Walls"
#       },
#       {
#         "id": 1,
#         "name": "Marla Spears"
#       },
#       {
#         "id": 2,
#         "name": "Roslyn Sutton"
#       }
#     ],
#     "greeting": "Hello, Tiffany Gonzalez! You have 3 unread messages.",
#     "favoriteFruit": "banana"
#   }

# # print(type(li))
# # ini_list = json.dumps(li)

# # tic = timeit.default_timer()
# # # initializing string representation of a list 
# # # ini_list = str(li)
# # ast.literal_eval(ini_list)
# # toc = timeit.default_timer()
# # print("Normal",toc-tic)




# # convert to string
# input = json.dumps(li)

# tic = timeit.default_timer()
# print(type(input))
# # load to dict
# my_dict = json.loads(input) 
# toc = timeit.default_timer()
# print("JSON",toc-tic)
# print(type(my_dict))

# # # printing initialized string of list and its type 
# # print ("initial string", ini_list) 
# # print (type(ini_list)) 
  
# # # Converting string to list 
# # res = json.loads(ini_list) 
  
# # # printing final result and its type 
# # print ("final list", res) 
# # print (type(res)) 