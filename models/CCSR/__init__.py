from . import inference_ccsr
from . import inference_ccsr_tile
from typing import List, Tuple, Optional
import os
import math
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf
from torch.nn import functional as F
from safetensors.torch import load_file
from ldm.xformers_state import disable_xformers
from model.q_sampler import SpacedSampler
from model.ccsr_stage1 import ControlLDM
from utils.image import auto_resize, pad
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts



def model_loader(config_path: str, ckpt_path: str, device: str) -> ControlLDM:
    model: ControlLDM = instantiate_from_config(OmegaConf.load(config_path))
    load_state_dict(model, torch.load(ckpt_path, map_location="cpu"), strict=True)


def process(
        model: ControlLDM,
        control_imgs: List[np.ndarray],
        steps: int,
        t_max: float,
        t_min: float,
        strength: float,
        color_fix_type: str,
        tiled: bool,
        tile_size: int,
        tile_stride: int
) -> Tuple[List[np.ndarray]]:
    """
    Apply CCSR model on a list of low-quality images.

    Args:
        model (ControlLDM): Model.
        control_imgs (List[np.ndarray]): A list of low-quality images (HWC, RGB, range in [0, 255]).
        steps (int): Sampling steps.
        t_max (float): The starting point of uniform sampling strategy.
        t_min (float): The ending point of uniform sampling strategy.
        strength (float): Control strength. Set to 1.0 during training.
        color_fix_type (str): Type of color correction for samples.
        tiled (bool): If specified, a patch-based sampling strategy will be used for sampling.
        tile_size (int): Size of patch.
        tile_stride (int): Stride of sliding patch.

    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
    """
    n_samples = len(control_imgs)
    sampler = SpacedSampler(model, var_type="fixed_small")
    control = torch.tensor(np.stack(control_imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()

    model.control_scales = [strength] * 13

    height, width = control.size(-2), control.size(-1)
    shape = (n_samples, 4, height // 8, width // 8)
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
    if not tiled:
        # samples = sampler.sample_ccsr_stage1(
        #     steps=steps, t_max=t_max, shape=shape, cond_img=control,
        #     positive_prompt="", negative_prompt="", x_T=x_T,
        #     cfg_scale=1.0, color_fix_type=color_fix_type
        # )
        samples = sampler.sample_ccsr(
            steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, color_fix_type=color_fix_type
        )
    else:
        samples = sampler.sample_with_mixdiff_ccsr(
            tile_size=tile_size, tile_stride=tile_stride,
            steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, color_fix_type=color_fix_type
        )

    x_samples = samples.clamp(0, 1)
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    control = (einops.rearrange(control, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)

    preds = [x_samples[i] for i in range(n_samples)]

    return preds


@torch.no_grad()
def process_tiled(
        model: ControlLDM,
        control_imgs: List[np.ndarray],
        steps: int,
        t_max: float,
        t_min: float,
        strength: float,
        color_fix_type: str,
        tile_diffusion: bool,
        tile_diffusion_size: int,
        tile_diffusion_stride: int,
        tile_vae: bool,
        vae_decoder_tile_size: int,
        vae_encoder_tile_size: int
) -> Tuple[List[np.ndarray]]:
    """
    Apply CCSR model on a list of low-quality images.

    Args:
        model (ControlLDM): Model.
        control_imgs (List[np.ndarray]): A list of low-quality images (HWC, RGB, range in [0, 255]).
        steps (int): Sampling steps.
        t_max (float): The starting point of uniform sampling strategy.
        t_min (float): The ending point of uniform sampling strategy.
        strength (float): Control strength. Set to 1.0 during training.
        color_fix_type (str): Type of color correction for samples.
        tile_diffusion (bool): If specified, a patch-based sampling strategy for diffusion peocess will be used for sampling.
        tile_diffusion_size (int): Size of patch for diffusion peocess.
        tile_diffusion_stride (int): Stride of sliding patch for diffusion peocess.
        tile_vae (bool): If specified, a patch-based sampling strategy for the encoder and decoder in VAE will be used.
        vae_decoder_tile_size (int): Size of patch for VAE decoder.
        vae_encoder_tile_size (int): Size of patch for VAE encoder.

    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
    """

    n_samples = len(control_imgs)
    sampler = SpacedSampler(model, var_type="fixed_small")
    # control = torch.tensor(np.stack(control_imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = torch.tensor(np.stack(control_imgs) / 255.0, dtype=model.dtype, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()

    empty_text_embed_sd = load_file(os.path.join(os.path.dirname(__file__), "empty_text_embed.safetensors"))
    empty_text_embed = empty_text_embed_sd['empty_text_embed'].to(model.dtype).to(model.device)

    model.control_scales = [strength] * 13
    height, width = control.size(-2), control.size(-1)
    shape = (n_samples, 4, height // 8, width // 8)
    # x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
    x_T = torch.randn(shape, device=model.device, dtype=model.dtype)
    if not tile_diffusion and not tile_vae:
        # samples = sampler.sample_ccsr_stage1(
        #     steps=steps, t_max=t_max, shape=shape, cond_img=control,
        #     positive_prompt="", negative_prompt="", x_T=x_T,
        #     cfg_scale=1.0,
        #     color_fix_type=color_fix_type
        # )
        samples = sampler.sample_ccsr(empty_text_embed=empty_text_embed,
            steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=8.0,
            color_fix_type=color_fix_type
        )
    else:
        if tile_vae:
            model._init_tiled_vae(encoder_tile_size=vae_encoder_tile_size, decoder_tile_size=vae_decoder_tile_size)
        if tile_diffusion:
            samples = sampler.sample_with_tile_ccsr(empty_text_embed=empty_text_embed,
                tile_size=tile_diffusion_size, tile_stride=tile_diffusion_stride,
                steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=control,
                positive_prompt="", negative_prompt="", x_T=x_T,
                cfg_scale=8.0,
                color_fix_type=color_fix_type
            )
        else:
            samples = sampler.sample_ccsr(empty_text_embed=empty_text_embed,
                steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=control,
                positive_prompt="", negative_prompt="", x_T=x_T,
                cfg_scale=1.0,
                color_fix_type=color_fix_type
            )

    x_samples = samples.clamp(0, 1)
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)

    preds = [x_samples[i] for i in range(n_samples)]

    return preds
