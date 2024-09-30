import PIL
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
        self.load_image()

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
        save_path = os.path.join(path, os.path.basename(os.path.dirname(self.img_path)), self.name)
        self.img_data.save(save_path)