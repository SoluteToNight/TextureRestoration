from .node import Node
from img_class import TextureImage as timg
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionImg2ImgPipeline
# from modules import mixdiff

class Diffusion(Node):
    def __init__(self, inputs: list[timg] = None):
        super().__init__(inputs)
        self.model = "models/architecturerealmix"
        self.convert()

    def convert(self):
        for img in self.img_list:
            img.tmp_data = img.img_data
            img.tmp_data = img.tmp_data.convert("RGB")

    def process(self, *args):
        if args is not None and args[0] == False:
            model_id = self.model
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16,
                                                                  safety_checker=None)
            pipe.to("cuda")
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            prompt = "high quality image"  # "deblur and erase the distortion"
            neg_prompt = "blur,distortion,shadow"
            for img in self.img_list:
                image = img.tmp_data
                images = pipe(prompt, negtive_prompt=neg_prompt, image=image, strength=0.20, num_inference_steps=45).images[0]
                img.tmp_data = images
                img.update()
            return
        elif args[0]:
            return
