from .node import Node
import numpy as np
import cv2 as cv
import os


# 蒙版
class Masking(Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
        print("Masking")
    def convert(self):
        for img in self.img_list:
            img.tmp_data = np.array(img.img_data)[:, :, ::-1]
            # cv.cvtColor(img.tmp_data, cv.COLOR_BGR2HSV)
    def process(self):
        self.convert()
        for img in self.img_list:
            blue_low = np.array([210,0,0])
            blue_high = np.array([255,0,0])
            # blue_low = cv.cvtColor(blue_low, cv.COLOR_RGB2HSV)
            # blue_high = cv.cvtColor(blue_high, cv.COLOR_RGB2HSV)
            mask = cv.inRange(img.tmp_data, blue_low, blue_high)
            mask_path = os.path.join(img.building_obj.temp_path,img.name)
            print(mask_path)
            mask_path = mask_path.replace(".png","_mask.png")
            cv.imwrite(mask_path, mask)
    def convert_back(self):
        for img in self.img_list:
            img.tmp_data = None
