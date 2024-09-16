import os.path

import cv2 as cv
from . import node
import numpy as np
import matplotlib



class image:
    def __init__(self,path):
        self.data = cv.imread(path)
        self.name = os.path.basename(path)
        self.brightness = None
    def save(self, path):
        cv.imwrite(path, self.data)


class Brightness(node.Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
        self.img_list = []
        for path in inputs:
            self.img_list.append(image(path))

    def process(self):
        overall_brightness = self.calculate_brightness()
        average_brightness = overall_brightness / len(self.img_list)
        low = min(self.img_list,key=lambda x: x.brightness)
        high = max(self.img_list,key=lambda x: x.brightness)
        print("Average brightness:", average_brightness)
        if average_brightness - low.brightness > 10 or high.brightness - average_brightness > 10:
            if average_brightness - low.brightness > high.brightness - average_brightness:
                print("Brightness is too low")
                value = average_brightness- low.brightness
                self.balance_brightness(low,value)
                return self.process()
            else:
                print("Brightness is too high")
                value = average_brightness-high.brightness
                self.balance_brightness(high,value)
                return self.process()
        else:
            print("Brightness is OK")
            return self.img_list

    def calculate_brightness(self):
        overall_brightness = 0
        for img in self.img_list:
            data = img.data
            brightness_weights = [0.114, 0.587, 0.299]
            weighted_brightness = np.sum(data[..., :3] * np.array(brightness_weights).reshape(1, 1, 3), axis=2)
            brightness = weighted_brightness.mean()
            print(img.name, ":", brightness)
            img.brightness = brightness
            overall_brightness += brightness
        return overall_brightness
    def balance_brightness(self,img:image,value):
        data = img.data.astype(np.float32)
        brightness_weights = np.array([0.114, 0.587, 0.299])
        weighted_brightness = np.sum(img.data[..., :3] * brightness_weights.reshape(1, 1, 3), axis=2, keepdims=True)
        adjusted_brightness = weighted_brightness * (1+value/255)
        delta_brightness = adjusted_brightness - weighted_brightness
        adjusted_img = img.data + delta_brightness
        adjusted_img = np.clip(adjusted_img, 0, 255)
        # 转换回uint8类型
        data = adjusted_img.astype(np.uint8)
        img.data = data
