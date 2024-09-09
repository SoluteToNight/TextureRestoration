from . import node
import ollama
import base64
from io import BytesIO
from PIL import Image

#调用llava进行分析
def convert_to_base64(pil_image):
    # 将PIL图像转换为字节流
    buffered = BytesIO()
    pil_image = pil_image.convert("RGB")
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def load_image(path):
    image = Image.open(path)
    return convert_to_base64(image)


class Analyse(node.Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
        self.model = "llava:latest"
        self.keywords = ["blur,missing,watermark,color balance,noise,lighting"]
    def process(self):
        image = load_image(self.input)
        message = {
            'role': 'user',
            'content': f'This is a picture of a modern building,but the quality of this picture is low.Please analyse the proplem of this picture,please give me answer base on these key words{self.keywords}',
            'images': [image]
        }
        stream = ollama.chat(
            model=self.model,
            messages=[message],
            stream=True
        )
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
        self.output = stream
        return self.output
