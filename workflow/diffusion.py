from .node import Node
from img_class import TextureImage as timg
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


class Diffusion(Node):
    def __init__(self, inputs: list[timg] = None):
        super().__init__(inputs)
        self.model = "models/instruct-pix2pix"
        self.convert()

    def convert(self):
        for img in self.img_list:
            img.tmp_data = img.img_data
            img.tmp_data = img.tmp_data.convert("RGB")

    def process(self, *args):
        model_id = self.model
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float32,
                                                                      safety_checker=None)
        pipe.to("cuda")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        prompt = "high quality image" # "deblur and erase the distortion"
        # neg_prompt = "blur,distortion,shadow"
        for img in self.img_list:
            image = img.tmp_data
            images = pipe(prompt,  image=image, num_inference_steps=45, image_guidance_scale=1.5).images[0]
            img.tmp_data = images
            img.update()
        return
