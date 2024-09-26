from PIL import Image
class TextureImage:
    def __init__(self, path):
        self.img_path = path
        self.img_data = Image.open(path)
        self.tmp_data = None

    def save(self, path):
        save_path = os.path.join(path, os.path.basename(self.img_path))
        self.img_data.save(save_path)

    def update(self):
        self.img_data = self.tmp_data
        self.tmp_data = None