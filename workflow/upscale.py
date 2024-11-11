import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
import pytorch_lightning as pl
from .node import Node
import safetensors
from safetensors.torch import load_file
current_dir = os.path.dirname(os.path.abspath(__file__))
ccsr_path = os.path.join(current_dir, '../models/CCSR')
sys.path.append(ccsr_path)
from models.CCSR import instantiate_from_config, load_state_dict, ControlLDM, auto_resize
import models.CCSR as CCSR


class Upscale(Node):
    def __init__(self, inputs=None, scale=2):
        super().__init__(inputs)
        self.scale = scale
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pl.seed_everything(123)
        config_file = os.path.join(current_dir, '../models/CCSR/configs/model/ccsr_stage2.yaml')
        # model_path = os.path.join(current_dir, '../models/CCSR/model/real-world_ccsr-fp32.ckpt')
        model_path = os.path.join(current_dir, '../models/CCSR/model/real-world_ccsr-fp16.safetensors')
        self.model: ControlLDM = instantiate_from_config(OmegaConf.load(config_file))
        load_state_dict(self.model, load_file(model_path))
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def process(self,
                sr_scale=1,
                steps: int = 20,
                t_max: float = 0.6667,
                t_min: float = 0.3333,
                strength: float = 1.0,
                color_fix_type: str = "adain",
                tiled: bool = True,
                tile_size: int = 512,
                tile_stride: int = 256):
        for img in self.img_list:
            lq = img.img_data
            if sr_scale != 1:
                lq = lq.resize(
                    tuple(math.ceil(x * sr_scale) for x in lq.size),
                    Image.Resampling.BICUBIC
                )
            if not tiled:
                lq_resized = auto_resize(lq, 512)
            else:
                lq_resized = auto_resize(lq, tile_size)

            x = lq_resized.resize(
                tuple(s // 64 * 64 for s in lq_resized.size), Image.Resampling.LANCZOS
            )
            x = np.array(x)
            # x = pad(np.array(lq_resized), scale=64)
            # preds = CCSR.process(
            #     self.model, [x], steps=steps,
            #     t_max=t_max, t_min=t_min,
            #     strength=1,
            #     color_fix_type=color_fix_type,
            #     tiled=True, tile_size=tile_size, tile_stride=tile_stride
            # )
            preds = CCSR.process_tiled(
                self.model, [x], steps=steps,
                t_max=t_max, t_min=t_min,
                strength=strength,
                color_fix_type=color_fix_type,
                tile_diffusion=True, tile_diffusion_size=512, tile_diffusion_stride=256,
                tile_vae=False, vae_encoder_tile_size=1024, vae_decoder_tile_size=224
            )
            pred = preds[0]
            img.img_data = Image.fromarray(pred).resize(lq.size, Image.Resampling.LANCZOS)
        del self.model
        return