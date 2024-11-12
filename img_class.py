import PIL
import numpy as np
import cv2 as cv
from PIL import Image
import os
import requests


class TextureImage:
    def __init__(self, path: str):
        self.img_path: str = path
        self.img_data = None
        self.tmp_data = None
        self.brightness = None
        self.name = os.path.basename(path)
        self.building_obj = None
        self.load_image()
        print(self)
    def __str__(self):
        return f"TextureImage loaded from:{self.img_path}"
    def load_image(self):
        if isinstance(self.img_path, str):
            if self.img_path.startswith("http://") or self.img_path.startswith("https://"):
                try:
                    response = requests.get(self.img_path, stream=True)
                    response.raise_for_status()
                    image = Image.open(response.raw)
                    image.convert("RGB")
                    self.img_data = image
                    return
                except requests.exceptions.HTTPError as e:
                    print(f"HTTP error occurred: {e}")
            elif os.path.isfile(self.img_path):
                image = Image.open(self.img_path)
                image.convert("RGB")
                self.img_data = image
                return
            else:
                raise FileNotFoundError(f"Incorrect path or URL")

    def update(self):
        if isinstance(self.tmp_data, Image.Image):
            self.img_data = self.tmp_data
        else:
            raise TypeError("tmp_data is not an PIL image,check convert_back")
        self.tmp_data = None

    def save(self, path):
        save_path = os.path.join(path, self.name)
        self.img_data.save(save_path)

    def tmp_save(self, data: [np.ndarray] = None):
        tmp_path = os.path.join(self.building_obj.temp_path, self.name)
        if data is None:
            self.img_data.save(tmp_path)
        elif isinstance(data, np.ndarray):
            self.tmp_data = Image.fromarray(data)
            self.tmp_data.save(tmp_path)
    def covert2nparray(self):
        cv_img = np.array(self.img_data)[:,:,::-1]  # BGR格式
        self.tmp_data = cv_img
        return cv_img
    def reconvert2PIL(self):
        img = cv.cvtColor(self.tmp_data,cv.COLOR_BGR2RGB)
        self.update()
        return Image.fromarray(img)