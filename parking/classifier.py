import matplotlib.pyplot as plt

import numpy as np
import torch
import torchvision
from torchvision import models
from torch import nn
from torch import optim
import cv2

def predict(model, nonProcessedImage):
    image = processImage(nonProcessedImage)
    output = model.cpu().forward(image)
    # print(output)
    output = torch.exp(output)
    pollings, classes = output.topk(1, dim=1)
    prob = output[0][1]/(output[0][1] + output[0][0])
    # print(pollings.item(), ' ', classes.item())
    classe = classes.item()
    # if prob < 0.95:
    #     classe = 0
    # else:
    #     classe = 1
    return classe

def processImage(img):
    # Get the dimensions of the image
    # width, height = img.size
    
    # Resize by keeping the aspect ratio, but changing the dimension
    # so the shortest size is 255px
    # img = img.resize((150, int(150*(height/width))) if width < height else (int(150*(width/height)), 150))

    # img = cv2.resize(img, (150, 150))
    
    # Get the dimensions of the new image size
    # width, height = img.size
    
    # Set the coordinates to do a center crop of 224 x 224
    # left = (width - 224)/2
    # top = (height - 224)/2
    # right = (width + 224)/2
    # bottom = (height + 224)/2
    # img = img.crop((left, top, right, bottom))
    
    # Turn image into numpy array
    img = np.array(img)
    
    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))
    
    # Make all values between 0 and 1
    img = img/255
    
    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    
    # Add a fourth dimension to the beginning to indicate batch size
    img = img[np.newaxis,:]
    
    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    return image

def defineDenseNet() :
  # Get pretrained model using torchvision.models as models library
  model = models.densenet161(pretrained=False)
  # Turn off training for their parameters
  for param in model.parameters():
      param.requires_grad = False

  # Create new classifier for model using torch.nn as nn library
  classifier_input = model.classifier.in_features
  num_labels = 2 #PUT IN THE NUMBER OF LABELS IN YOUR DATA
  classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                             nn.ReLU(),
                             nn.Linear(1024, 512),
                             nn.ReLU(),
                             nn.Linear(512, num_labels),
                             nn.LogSoftmax(dim=1))
  # Replace default classifier with new classifier
  model.classifier = classifier
  
  return model

def defineOptimizer(model) :
  # Set the error function using torch.nn as nn library
  criterion = nn.NLLLoss()
  # Set the optimizer function using torch.optim as optim library
  optimizer = optim.Adam(model.classifier.parameters())
  return optimizer, criterion

def load_checkpoint(PATH):
  model = defineDenseNet()
  optimizer, criterion = defineOptimizer(model)

  checkpoint = torch.load(PATH, map_location = 'cpu')
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  print('Loss {:.6f}'.format(checkpoint['loss']))

  model.train()
  return model, optimizer, criterion