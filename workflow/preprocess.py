from .node import Node
import os
import shutil
# 预处理，虽然我也不知道干嘛用
from img_class import TextureImage as timg

class PreProcess(Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)

    def process(self,*args):
        input_path = args[0] if args else r"obj"
        output_path = args[1] if len(args) > 1 else r"outputs"
        scripts_path = os.getcwd()
        input_path = os.path.join(scripts_path, input_path)
        output_path = os.path.join(scripts_path, output_path)
        print(input_path)
        contents = os.listdir(input_path)
        folders = [content for content in contents if os.path.isdir(os.path.join(input_path, content))]
        print(folders)
        if not folders:   # 路径不包含文件夹，则直接处理文件
            self.mtl_handel(input_path)
            output_dir = os.path.join(output_path, os.path.basename(input_path))
            os.mkdir(output_dir)    # 输出文件夹下的同名文件夹
            for content in contents :
                if content.rfind(".obj") != -1 or content.rfind(".mtl") != -1:
                    in_path = os.path.join(input_path, content)
                    out_path = os.path.join(output_dir, content)
                    shutil.copy(in_path, out_path)
                elif content.rfind(".jpg") != -1 or content.rfind(".png") != -1:
                    img_path = os.path.join(input_path, content)
                    img = timg(img_path)
                    self.img_list.append(img)

    def mtl_handel(self,input_path):
        mtl_path = None
        files = os.listdir(input_path)
        for f in files:
            files_path = os.path.join(input_path,f)
            print(files_path)
            ext = files_path.rfind(".mtl")
            if ext != -1:
                mtl_path = files_path
                break
        print(mtl_path)
        dir_name = os.path.dirname(mtl_path)
        scripts_path = os.getcwd()
        os.chdir(dir_name)
        new_file = open("new.tmp", "w+")
        with open(os.path.basename(mtl_path), 'r', encoding='utf-8') as file:
            for line in file:
                if line.find("map_Kd") != -1:
                    route = line.split(" ", 1)
                    route = os.path.basename(route[1])
                    route = os.path.join("./", route)
                    line = "map_Kd " + route
                new_file.write(line)

        new_file.close()
        os.remove(os.path.basename(mtl_path))
        os.rename("new.tmp", os.path.basename(mtl_path))
        os.chdir(scripts_path)
