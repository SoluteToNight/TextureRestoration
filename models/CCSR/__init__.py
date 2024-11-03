import inference_ccsr
import inference_ccsr_tile
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
    control = torch.tensor(np.stack(control_imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()

    model.control_scales = [strength] * 13

    height, width = control.size(-2), control.size(-1)
    shape = (n_samples, 4, height // 8, width // 8)
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)

    if not tile_diffusion and not tile_vae:
        # samples = sampler.sample_ccsr_stage1(
        #     steps=steps, t_max=t_max, shape=shape, cond_img=control,
        #     positive_prompt="", negative_prompt="", x_T=x_T,
        #     cfg_scale=1.0,
        #     color_fix_type=color_fix_type
        # )
        samples = sampler.sample_ccsr(
            steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=8.0,
            color_fix_type=color_fix_type
        )
    else:
        if tile_vae:
            model._init_tiled_vae(encoder_tile_size=vae_encoder_tile_size, decoder_tile_size=vae_decoder_tile_size)
        if tile_diffusion:
            samples = sampler.sample_with_tile_ccsr(
                tile_size=tile_diffusion_size, tile_stride=tile_diffusion_stride,
                steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=control,
                positive_prompt="", negative_prompt="", x_T=x_T,
                cfg_scale=8.0,
                color_fix_type=color_fix_type
            )
        else:
            samples = sampler.sample_ccsr(
                steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=control,
                positive_prompt="", negative_prompt="", x_T=x_T,
                cfg_scale=1.0,
                color_fix_type=color_fix_type
            )

    x_samples = samples.clamp(0, 1)
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)

    preds = [x_samples[i] for i in range(n_samples)]

    return preds
# @torch.no_grad()
# def process(ccsr_model, image, resize_method, scale_by, steps, t_max, t_min, tile_size, tile_stride,
#             color_fix_type, keep_model_loaded, vae_tile_size_encode, vae_tile_size_decode, sampling_method, seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     dtype = ccsr_model['dtype']
#     model = ccsr_model['model']
#     device = "cuda"
#     # empty_text_embed = torch.load(os.path.join(script_directory, "empty_text_embed.pt"), map_location=device)
#     sampler = SpacedSampler(model, var_type="fixed_small")
#     width,height = image.size
#     image, = image.resize((scale_by*width, scale_by*height), Image.Resampling.BICUBIC)
#     B,H,W,C = image.shape
#     # Calculate the new height and width, rounding down to the nearest multiple of 64.
#     new_height = H // 64 * 64
#     new_width = W // 64 * 64
#
#     # Reorder to [B, C, H, W] before using interpolate.
#     image = image.permute(0, 3, 1, 2).contiguous()
#     resized_image = F.interpolate(image, size=(new_height, new_width), mode='bicubic', align_corners=False)
#
#     strength = 1.0
#     model.control_scales = [strength] * 13
#
#     model.to("cuda", dtype=dtype).eval()
#
#     height, width = resized_image.size(-2), resized_image.size(-1)
#     shape = (1, 4, height // 8, width // 8)
#     x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
#
#     out = []
#     if B > 1:
#         for i in range(B):
#             img = resized_image[i].unsqueeze(0).to(device)
#             if sampling_method == 'ccsr_tiled_mixdiff':
#                 model.reset_encoder_decoder()
#                 print("Using tiled mixdiff")
#                 samples = sampler.sample_with_mixdiff_ccsr(
#                     tile_size=tile_size, tile_stride=tile_stride,
#                     steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=img,
#                     positive_prompt="", negative_prompt="", x_T=x_T,
#                     cfg_scale=1.0,
#                     color_fix_type=color_fix_type
#                 )
#             elif sampling_method == 'ccsr_tiled_vae_gaussian_weights':
#                 model._init_tiled_vae(encoder_tile_size=vae_tile_size_encode // 8,
#                                       decoder_tile_size=vae_tile_size_decode // 8)
#                 print("Using gaussian weights")
#                 samples = sampler.sample_with_tile_ccsr(
#                      tile_size=tile_size, tile_stride=tile_stride,
#                     steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=img,
#                     positive_prompt="", negative_prompt="", x_T=x_T,
#                     cfg_scale=1.0,
#                     color_fix_type=color_fix_type
#                 )
#             else:
#                 model.reset_encoder_decoder()
#                 print("no tiling")
#                 samples = sampler.sample_ccsr(
#                     steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=img,
#                     positive_prompt="", negative_prompt="", x_T=x_T,
#                     cfg_scale=1.0,
#                     color_fix_type=color_fix_type
#                 )
#             out.append(samples.squeeze(0).cpu())
#             if B > 1:
#                 print("Sampled image ", i, " out of ", B)
#
#     original_height, original_width = H, W
#     processed_height = samples.size(2)
#     target_width = int(processed_height * (original_width / original_height))
#     out_stacked = torch.stack(out, dim=0).cpu().to(torch.float32).permute(0, 2, 3, 1)
#     resized_back_image, = ImageScale.upscale(self, out_stacked, "lanczos", target_width, processed_height,
#                                              crop="disabled")
#
#     if not keep_model_loaded:
#         model.to(offload_device)
#         mm.soft_empty_cache()
#     return (resized_back_image,)