import os
import shutil
# 预处理，虽然我也不知道干嘛用
from img_class import TextureImage as timg
def load_data(input_path: str = None, output_path: str = None):
    script_path = os.getcwd()
    if input_path is None:
        raise ValueError("input_path is None")
    if output_path is None:
        raise ValueError("output_path is None")
    if not os.path.isabs(input_path):
        os.path.abspath(input_path)
    if not os.path.isabs(output_path):
        os.path.abspath(output_path)
    if os.path.isfile(input_path):
        if input_path.endswith(".obj"):
            output_folder = create_output_folder(input_path, output_path)
            with open(input_path, "r") as f:
                while True:
                    content = f.readline()
                    if content.startswith("mtllib"):
                        os.chdir(os.path.dirname(input_path))
                        mtl_path = content.split(" ")[1]
                        mtl_handel(mtl_path)
                        print("mtl_path:", mtl_path)
                        shutil.copy(mtl_path, os.path.join(output_folder, mtl_path))
                        shutil.copy(input_path, os.path.join(output_folder, os.path.basename(input_path)))
                        os.chdir(script_path)
                        break
        elif os.path.isdir(input_path):
            for root, dirs, files in os.walk(input_path):
                # input_folder = root     # 按理来说这里应当是绝对路径
                if len(files) == 0:
                    continue
                for file in files:
                    if file.endswith(".obj"):
                        output_folder = create_output_folder(root, output_path)
                        obj_path = os.path.join(root, file)
                        with open(obj_path, "r") as f:
                            while True:
                                os.chdir(root)
                                content = f.readline()
                                if content.startswith("mtllib"):
                                    mtl_path = content.split(" ")[1]
                                    mtl_path = os.path.join(os.path.dirname(obj_path), mtl_path)
                                    mtl_handel(mtl_path)
                                    print("mtl_path:", mtl_path)
                                    shutil.copy(mtl_path, os.path.join(output_folder, mtl_path))
                                    shutil.copy(input_path, os.path.join(output_folder, os.path.basename(input_path)))
                                    os.chdir(script_path)
                                    break
        else:
            raise ValueError("input_path is not an obj file or a folder include an obj file")


def mtl_handel(mtl_path: str =None):
    print(mtl_path)
    dir_name = os.path.dirname(mtl_path)
    new_file = open("new.tmp", "w+")
    with open(mtl_path, 'r', encoding='utf-8') as file:
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


def create_output_folder(input_path:str = None,output_path: str = None):
    folder_path = os.path.basename(input_path)
    output_folder = os.path.join(output_path,folder_path)
    print(output_folder)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)
    with open(os.path.join(output_folder, "log.txt"), "w+") as f:
        f.write(f"Folder originate from {input_path}\n")
    return output_folder
