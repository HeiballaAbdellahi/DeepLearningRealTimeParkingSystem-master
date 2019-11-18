import cv2
from PIL import Image
import numpy as np


def extractCrop(image, slot):
	x = slot.x
	y = slot.y
	w = slot.width
	h = slot.height
	return image[y:y+h, x:x+w]

def imagefromArray(image):
	return Image.fromarray(image)

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)