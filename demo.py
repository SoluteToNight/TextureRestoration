import os
import workflow
import torch
from argparse import ArgumentParser, Namespace
from PIL import Image
from img_class import TextureImage as timg
from DataLoader import load_data

def arg_parser() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input_dir", default="obj/test/scene.obj", help="input image dir")
    parser.add_argument("--output_dir", default="outputs", help="output image dir")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--tile", default=False, action="store_true",help="tile")
    parser.add_argument("--tiling_size", default=512, help="tiling size")
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
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    args = arg_parser()
    script_path = os.getcwd()
    input_path = args.input_dir
    output_path = args.output_dir
    device = check_device(args.device)
    tile = args.tile
    tile_size = args.tiling_size
    print("device:", device)

    img_list = []
    building = load_data(input_path, output_path)
    try:
        bd = next(building)
        bd.load_texture()
        img_list = bd.texture_list
        for img in img_list:
            print(img)
        # workflow.PreProcess(img_list).process(input_path, output_path)
        # workflow.Analyse(img_list).process()
        # workflow.CCSR(img_list).process()
        workflow.Brightness(img_list).process()
        # workflow.Diffusion(img_list).process(tile, tile_size)
        # workflow.Upscale(img_list).process()
        workflow.Masking(img_list).process()
        for img in img_list:
            img.save(output_path)
    except StopIteration:
        print("No more data")

main()
