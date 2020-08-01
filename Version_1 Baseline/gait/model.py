from PIL import Image
import torch
from imutils import paths
import random
import numpy as np
import torch.nn as nn

class SiaCNN(nn.Module):
  def __init__(self):
    super(SiaCNN, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),      
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU())      
    self.avgp = nn.AvgPool2d(14)
    
  def forward(self, x):
    x=x.float()
    out = self.layer1(x)
    # print(out.shape)
    out = self.avgp(out)
    out = out.reshape(out.size(0), -1)
    # print(out.shape)
    return out