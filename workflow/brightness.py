import os.path
from PIL import Image
import cv2 as cv
from .node import Node
import numpy as np
from img_class import TextureImage as timg


class Brightness(Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
        self.convert()

    def convert(self):
        for img in self.img_list:
            img.tmp_data = np.array(img.img_data)[:, :, ::-1]

    def process(self):
        overall_brightness = self.calculate_brightness()
        average_brightness = overall_brightness / len(self.img_list)
        low = min(self.img_list, key=lambda x: x.brightness)
        high = max(self.img_list, key=lambda x: x.brightness)
        print("Average brightness:", average_brightness)
        if average_brightness - low.brightness > 10 or high.brightness - average_brightness > 10:
            if average_brightness - low.brightness > high.brightness - average_brightness:
                print("Brightness is too low")
                value = average_brightness - low.brightness
                self.balance_brightness(low, value)
                return self.process()
            else:
                print("Brightness is too high")
                value = average_brightness - high.brightness
                self.balance_brightness(high, value)
                return self.process()
        else:
            print("Brightness is OK")
            self.convert_back()
            return self.img_list

    def calculate_brightness(self):
        overall_brightness = 0
        for img in self.img_list:
            data = img.tmp_data
            brightness_weights = [0.114, 0.587, 0.299]
            weighted_brightness = np.sum(data[..., :3] * np.array(brightness_weights).reshape(1, 1, 3), axis=2)
            brightness = weighted_brightness.mean()
            print(img.name, ":", brightness)
            img.brightness = brightness
            overall_brightness += brightness
        return overall_brightness

    def balance_brightness(self, img, value):
        data = img.tmp_data.astype(np.float32)
        brightness_weights = np.array([0.114, 0.587, 0.299])
        weighted_brightness = np.sum(data[..., :3] * brightness_weights.reshape(1, 1, 3), axis=2, keepdims=True)
        adjusted_brightness = weighted_brightness * (1 + value / 255)
        delta_brightness = adjusted_brightness - weighted_brightness
        adjusted_img = data + delta_brightness
        adjusted_img = np.clip(adjusted_img, 0, 255)
        # 转换回uint8类型
        data = adjusted_img.astype(np.uint8)
        img.tmp_data = data

    def convert_back(self):
        for img in self.img_list:
            img.tmp_data = cv.cvtColor(img.tmp_data, cv.COLOR_BGR2RGB)
            img.tmp_data = Image.fromarray(img.tmp_data)
            img.update()
