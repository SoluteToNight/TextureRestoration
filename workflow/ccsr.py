import torch
import os
import math
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import pytorch_lightning as pl
from omegaconf import OmegaConf
from .node import Node

current_dir = os.path.dirname(os.path.abspath(__file__))
ccsr_path = os.path.join(current_dir, '../models/CCSR')
sys.path.append(ccsr_path)

from models.CCSR.model.q_sampler import SpacedSampler
from models.CCSR.model.ccsr_stage1 import ControlLDM
from models.CCSR.utils.image import auto_resize
from models.CCSR.utils.common import instantiate_from_config, load_state_dict


class CCSR(Node):
    def __init__(self, inputs=None, scale=2):
        super().__init__(inputs)
        self.scale = scale
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        config_file = os.path.join(current_dir, '../models/CCSR/configs/model/ccsr_stage2.yaml')
        self.model = instantiate_from_config(OmegaConf.load(config_file))

        model_path = os.path.join(current_dir, '../models/CCSR/model/real-world_ccsr-fp32.ckpt')
        try:
            load_state_dict(
                self.model,
                torch.load(model_path, map_location=self.device),
                strict=True
            )
            print(f"模型权重加载成功: {model_path}")
        except Exception as e:
            print(f"模型权重加载失败: {e}")
            return

        self.model.eval().cuda()
        # self.model.to(self.device)

    # @torch.no_grad()
    def process(self):
        for img in self.img_list:
            if img.img_data is None:
                raise ValueError(f"Image {img.name} has no tmp_data for CCSR process")
            if not isinstance(img.img_data, Image.Image):
                raise ValueError(f"Invalid tmp_data for image {img.name} in CCSR process")
            img_data = img.img_data.convert("RGB")
            img_np = np.array(img_data)

            lq = Image.fromarray(img_np)
            if self.scale != 1:
                lq = lq.resize(
                    tuple(math.ceil(x * self.scale) for x in lq.size), Image.Resampling.BICUBIC
                )
            lq_resized = auto_resize(lq, 512)
            x = lq_resized.resize(tuple(s // 64 * 64 for s in lq_resized.size), Image.Resampling.LANCZOS)
            x = np.array(x)

            sr_img_np = self.run_ccsr(x)

            sr_img = Image.fromarray(sr_img_np)

            img.img_data = sr_img
            # img.update()

        return self.img_list

    def run_ccsr(self, img_np):
        print("开始推理...")
        with torch.no_grad():
            control = torch.tensor(np.stack([img_np]) / 255.0, dtype=torch.float32, device=self.device).clamp_(0, 1)
            control = control.permute(0, 3, 1, 2).contiguous()
            print("输入图像处理完成")

            sampler = SpacedSampler(self.model, var_type="fixed_small")
            shape = (1, 4, control.size(-2) // 8, control.size(-1) // 8)
            x_T = torch.randn(shape, device=self.device, dtype=torch.float32)
            print(f"生成随机张量 x_T: {x_T.shape}")

            steps = 20
            with tqdm(total=steps, desc="推理进度", unit="step") as pbar:
                for step in range(steps):
                    samples = sampler.sample_ccsr(
                        steps=steps,
                        t_max=0.6667,
                        t_min=0.3333,
                        shape=shape,
                        cond_img=control,
                        positive_prompt="",
                        negative_prompt="",
                        x_T=x_T,
                        cfg_scale=1.0,
                        color_fix_type="adain",
                    )
                    pbar.update(1)

            print("推理完成，开始生成图像")
            sr_img_np = (samples.clamp(0, 1).permute(0, 2, 3, 1) * 255).cpu().numpy().astype(np.uint8)[0]
            print("图像生成完成")

        return sr_img_np
