import os
import cv2
import torch
import numpy as np
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from models.LISA import LISAForCausalLM
from models.LISA.model.segment_anything.utils.transforms import ResizeLongestSide
from .node import Node
from models.LISA.model.llava.mm_utils import tokenizer_image_token
from models.LISA.model.llava import conversation as conversation_lib
from models.LISA.utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


class Masking(Node):
    def __init__(self, img_list, lisa_version="models/LISA-7B-v1", precision="fp16", load_in_4bit=True,
                 load_in_8bit=False, image_size=1024, save_path="./vis_output", use_mm_start_end=True):
        super().__init__(img_list)
        self.img_list = img_list
        self.image_size = image_size
        self.precision = precision
        self.save_path = save_path
        self.use_mm_start_end = use_mm_start_end
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            lisa_version,
            model_max_length=512,
            padding_side="right",
            use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

        torch_dtype = torch.float32
        if precision == "bf16":
            torch_dtype = torch.bfloat16
        elif precision == "fp16":
            torch_dtype = torch.half

        kwargs = {"torch_dtype": torch_dtype}
        if load_in_4bit:
            kwargs.update({
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            })
        elif load_in_8bit:
            kwargs.update({
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True),
            })

        self.model = LISAForCausalLM.from_pretrained(
            lisa_version, low_cpu_mem_usage=True,vision_tower="models/clip-vit-large-patch14",  seg_token_idx=self.seg_token_idx, **kwargs)
        #self.model.eval()
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # 初始化视觉模块
        self.model.get_model().initialize_vision_modules(self.model.get_model().config)
        #self.clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        #self.transform = ResizeLongestSide(self.image_size)
        vision_tower = self.model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch_dtype)

        if self.precision == "bf16":
            self.model = self.model.bfloat16().cuda()
        elif self.precision == "fp16" and (not load_in_4bit) and (not load_in_8bit):
            vision_tower = self.model.get_model().get_vision_tower()
            self.model.model.vision_tower = None
            import deepspeed

            model_engine = deepspeed.init_inference(
                model=self.model,
                dtype=torch.half,
                replace_with_kernel_inject=True,
                replace_method="auto",
            )
            self.model = model_engine.module
            self.model.model.vision_tower = vision_tower.half().cuda()
        elif self.precision == "fp32":
            self.model = self.model.float().cuda()

        vision_tower = self.model.get_model().get_vision_tower()
        vision_tower.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.clip_image_processor = CLIPImageProcessor.from_pretrained("models/clip-vit-large-patch14")
        self.transform = ResizeLongestSide(self.image_size)

        self.model.eval()
        #if torch.cuda.is_available():
        #    self.model = self.model.cuda()



    def preprocess_image(self, image_np):
        image_clip = (
            self.clip_image_processor.preprocess(images=image_np, return_tensors="pt")[
            "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
        if self.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif self.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        processed_image = self.transform.apply_image(image_np)
        resize_list = [processed_image.shape[:2]]
        original_size_list = [image_np.shape[:2]]

        image_tensor = (
            self.preprocess(torch.from_numpy(processed_image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )

        if self.precision == "bf16":
            image_tensor = image_tensor.bfloat16()
        elif self.precision == "fp16":
            image_tensor = image_tensor.half()
        else:
            image_tensor = image_tensor.float()

        return image_clip, image_tensor, [processed_image.shape[:2]], [image_np.shape[:2]]

    def preprocess(self, x):
        pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).cuda()
        pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).cuda()
        x = x.cuda()
        x = (x - pixel_mean) / pixel_std
        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        x = torch.nn.functional.pad(x, (0, padw, 0, padh))
        return x

    def generate_mask_prompt(self, prompt: str, image_np):
        conv = conversation_lib.conv_templates["llava_v1"].copy()
        conv.messages = []

        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if self.use_mm_start_end:
            replace_token = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
        image_clip, image_tensor, resize_list, original_size_list = self.preprocess_image(image_np)

        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        print(f"image_clip shape before evaluate: {image_clip.shape}")
        print(f"image_tensor shape before evaluate: {image_tensor.shape}")

        with torch.no_grad():
            output_ids, pred_masks = self.model.evaluate(
                image_clip,
                image_tensor,
                input_ids,
                resize_list,
                original_size_list,
                max_new_tokens=512,
                tokenizer=self.tokenizer
            )


        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        text_output = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        print("LISA Response:", text_output)

        return pred_masks

    def process(self):
        prompt = input("请输入蒙版处理相关的提示: ")

        for img in self.img_list:

            image_np = np.array(img.img_data)
            print("Image shape:", image_np.shape)
            print("Image dtype:", image_np.dtype)



            pred_masks = self.generate_mask_prompt(prompt, image_np)


            for i, pred_mask in enumerate(pred_masks):
                if pred_mask.shape[0] == 0:
                    continue

                pred_mask = pred_mask.detach().cpu().numpy()[0]
                pred_mask = (pred_mask > 0).astype(np.uint8) * 255
                save_path = os.path.join(self.save_path, f"{img.name}_mask_{i}.png")
                cv2.imwrite(save_path, pred_mask)
                print(f"{save_path} 已保存。")


                save_img = image_np.copy()
                save_img[pred_mask > 0] = (save_img[pred_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
                masked_img_path = os.path.join(self.save_path, f"{img.name}_masked_img_{i}.png")
                cv2.imwrite(masked_img_path, cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))
                print(f"{masked_img_path} 已保存。")


        for img in self.img_list:
            img.tmp_data = None

