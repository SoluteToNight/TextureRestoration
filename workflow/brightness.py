import cv2 as cv
from . import node
import numpy as np
import matplotlib


img_list = []
def read_image(path):
    img = cv.imread(path)
    img_list.append(cv.imread(path))
class Brightness(node):
    def __init__(self,inputs=None):
        super().__init__(inputs)
        for path in inputs:
            read_image(path)
    def process(self):
        for img in img_list:
            brightness_weights = [0.114, 0.587, 0.299]
            weighted_brightness = np.sum(img[..., :3] * np.array(brightness_weights).reshape(1, 1, 3), axis=2)
            average_brightness = weighted_brightness.mean()
            print(average_brightness)
            # bright = 0.299 * R + 0.587 * G + 0.114 * B
