from .node import Node
import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline


class Upscale(Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
        self.model = "./models/Flux.1-dev-Controlnet-Upscaler"
    def process(self):
        # Load pipeline
        controlnet = FluxControlNetModel.from_pretrained(
            self.model,
            torch_dtype=torch.bfloat32
        )
        pipe = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            controlnet=controlnet,
            torch_dtype=torch.bfloat32
        )
        pipe.to("cuda")
        for img in self.img_list:
            control_image = img.img_data
            w, h = control_image.size
        # Upscale x4
            control_image = control_image.resize((w * 4, h * 4))
            image = pipe(
                prompt="",
                control_image=control_image,
                controlnet_conditioning_scale=0.6,
                num_inference_steps=28,
                guidance_scale=3.5,
                height=control_image.size[1],
                width=control_image.size[0]
            ).images[0]

