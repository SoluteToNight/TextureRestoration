from .node import Node
import ollama
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoModelForImageSegmentation, AutoImageProcessor, AutoModelForCausalLM

#调用llava进行分析

class Analyse(Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
        self.model = "llava:latest"
        self.lisa_model = AutoModelForImageSegmentation.from_pretrained("xinlai/LISA-7B-v1")
        self.processor = AutoImageProcessor.from_pretrained("xinlai/LISA-7B-v1")
        self.keywords = ["blur,missing,watermark,color balance,noise,lighting"]
        self.convert()
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
                'images': [img.tmp_data]
            }
            stream = ollama.chat(
                model=self.model,
                messages=[message],
                stream=True
            )
            for chunk in stream:
                print(chunk['message']['content'], end='', flush=True)
            #output = stream

        for img in self.img_list:
                # 预处理图像
            inputs = self.processor(images=img.img_data, return_tensors="pt")
            outputs = self.lisa_model(**inputs)

                # 生成分割结果
            segmentation_map = outputs.logits.argmax(-1).squeeze().cpu().numpy()

                # 将分割结果可视化并保存
            img.segmentation_result = segmentation_map
            img.update()


        return output
