from .node import Node
from img_class import TextureImage as timg
import torch
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

class Diffusion(Node):
    def __init__(self, inputs: list[timg] = None):
        super().__init__(inputs)
        self.model = "stablediffusionapi/architecturerealmix"
        self.convert()
        self.sam_model = sam_model_registry["vit_h"](checkpoint="models/segment-anything/segment_anything/sam_vit_h_4b8939.pth")
        self.sam_predictor = SamPredictor(self.sam_model)

    def convert(self):
        """ Convert images in the img_list to RGB format """
        for img in self.img_list:
            if img.img_data:
                img.tmp_data = img.img_data.convert("RGB")
            else:
                raise ValueError(f"Image data missing for {img}")

    def segment_with_sam(self):
        """ 使用 SAM 模型对图像进行全景分割 """
        for img in self.img_list:
            image = np.array(img.tmp_data)
            self.sam_predictor.set_image(image)

            # 生成分割掩码（可以根据具体需求修改）
            masks, _, _ = self.sam_predictor.predict()

            # 将掩码应用到图像上，这里你可以根据掩码对不同区域做进一步处理
            img.tmp_data = Image.fromarray((masks[0] * 255).astype(np.uint8))  # 将掩码保存为图像

    def process(self, *args):
        """ Process the images based on the provided args """
        if not self.img_list or args is None:
            raise ValueError("No images to process or no arguments provided")

        # Step 1: 先使用 SAM 进行分割
        self.segment_with_sam()

        # Step 2: 使用 Stable Diffusion 做进一步处理
        model_id = self.model

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None
        )
        pipe.to("cuda")

        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

        prompt = "high quality image"
        negative_prompt = "blur, distortion, shadow"

        for img in self.img_list:
            if img.tmp_data:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=img.tmp_data,
                    strength=0.20,
                    num_inference_steps=45
                ).images[0]

                img.tmp_data = result
                img.update()

        return
