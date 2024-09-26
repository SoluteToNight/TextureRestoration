from . import node
import ollama
import base64
from io import BytesIO
from PIL import Image

#调用llava进行分析

class Analyse(node.Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
        self.model = "llava:latest"
        self.keywords = ["blur,missing,watermark,color balance,noise,lighting"]
    def convert(self):
        for img in self.img_list:
            buffered = BytesIO()
            img.tmp_data = img.img_data.convert("RGB")
            img.tmp_data.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            img.tmp_data = img_str
    def process(self):
        for img in self.img_list:
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
            output = stream
            return output
