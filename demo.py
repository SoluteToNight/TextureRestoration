import os
import workflow
import torch
from argparse import ArgumentParser, Namespace
from PIL import Image
from img_class import TextureImage as timg


def arg_parser() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input_dir",default="obj/test",help="input image dir")
    parser.add_argument("--output_dir",default="outputs",help="output image dir")
    parser.add_argument("--device",default="cuda",help="device")
    return parser.parse_args()
def check_device(device):
    if device == "cpu":
        return "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            print("No CUDA device is available, using CPU instead")
            return "cpu"
        return "cuda"
def main() -> None:
    args = arg_parser()
    script_path = os.getcwd()
    input_path = args.input_dir
    output_path = args.output_dir
    device = check_device(args.device)
    print("device:",device)
    print(f"Loading image from{input_path}")
    img_list = []

    workflow.PreProcess(img_list).process(input_path, output_path)
    workflow.Analyse(img_list).process()
    workflow.Brightness(img_list).process()
    workflow.Diffusion(img_list).process()
    for img in img_list:
        img.save(output_path)

main()