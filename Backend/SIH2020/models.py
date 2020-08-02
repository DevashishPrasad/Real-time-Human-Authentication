from django.db import models

class BackendDictionary(models.Model):
    dicts = models.CharField(max_length=120000000000000)
    def __str__(self):
        return self.dicts

class Shared(models.Model):
    dicts = models.CharField(max_length=120000000000000)
    def __str__(self):
        return self.dicts

        
# from kora.models import Persona
# import cv2
# import base64
# image = cv2.imread("/home/ayan_gadpal/Pictures/a.png")

# ret, jpeg = cv2.imencode('.jpg', image)
# buffer1 = base64.b64encode(jpeg).decode('ascii')
# T = Persona(first_name="Maniya",image=buffer1)
# T.save()