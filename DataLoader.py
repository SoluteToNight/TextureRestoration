import os
import shutil
from img_class import TextureImage as timg
from BuildingObj import BuildingObj

def load_data(input_path: str = None, output_path: str = None):
    arg_list = []
    script_path = os.getcwd()
    # print(script_path)
    if input_path is None:
        raise ValueError("input_path is None")
    if output_path is None:
        raise ValueError("output_path is None")
    if not os.path.isabs(input_path):
        input_path = os.path.abspath(input_path)
    if not os.path.isabs(output_path):
        output_path = os.path.abspath(output_path)
    print(input_path, output_path)
    if os.path.isfile(input_path):
        if input_path.endswith(".obj"):
            print("input path is a single file")
            output_folder = create_output_folder(os.path.dirname(input_path), output_path)
            # print(output_folder)
            print(f"Loading images from {input_path}")
            with open(input_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("mtllib"):
                        os.chdir(os.path.dirname(input_path))
                        mtl_path = line.split(" ")[1]
                        mtl_path = mtl_path.replace("\n","")
                        mtl_handel(mtl_path)
                        shutil.copy(mtl_path, os.path.join(output_folder, mtl_path))
                        shutil.copy(input_path, os.path.join(output_folder, os.path.basename(os.path.basename(input_path))))
                        mtl_path = os.path.abspath(mtl_path)
                        os.chdir(script_path)
                        temp_path = create_temp_folder(output_folder)
                        temp_path = os.path.abspath(temp_path)
                        arg_list.append((input_path, mtl_path, temp_path))
                        break
    elif os.path.isdir(input_path):
            #当路径是文件夹时，处理所有obj
            print("input_path is a folder")
            for root, dirs, files in os.walk(input_path):
                # input_folder = root     # 按理来说这里应当是绝对路径
                if len(files) == 0:
                    continue
                for file in files:
                    if file.endswith(".obj"):
                        output_folder = create_output_folder(root, output_path)
                        print(f"output folder{output_folder}")
                        obj_path = os.path.join(root, file)
                        with open(obj_path, "r") as f:
                            while True:
                                os.chdir(root)
                                content = f.readline()
                                if content.startswith("mtllib"):
                                    mtl_path = content.split(" ")[1]
                                    mtl_path = os.path.join(os.path.dirname(obj_path), mtl_path)
                                    # 我不知道为什么这里会多一个换行符
                                    mtl_path = mtl_path.replace("\n", "")
                                    mtl_handel(mtl_path)
                                    print("mtl_path:", mtl_path)
                                    shutil.copy(mtl_path, os.path.join(output_folder, os.path.basename(mtl_path)))
                                    shutil.copy(obj_path, os.path.join(output_folder, os.path.basename(input_path)))
                                    mtl_path = os.path.abspath(mtl_path)
                                    os.chdir(script_path)
                                    temp_path = create_temp_folder(output_folder)
                                    temp_path = os.path.abspath(temp_path)
                                    arg_list.append((input_path, mtl_path, temp_path))
                                    break
                        break
    else:
        raise ValueError("input_path is not an obj file")
    return load_building(arg_list)
def load_building(arg_list):
    for obj_path,mtl_path,temp_path in arg_list:
        yield BuildingObj(obj_path,mtl_path,temp_path)

def mtl_handel(mtl_path: str =None):
    print(mtl_path)
    dir_name = os.path.dirname(mtl_path)
    new_file = open("new.tmp", "w+")
    with open(mtl_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.find("map_Kd") != -1:
                route = line.split(" ", 1)
                route = os.path.basename(route[1])
                # route = os.path.join("./", route)
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
def create_temp_folder(tree_path: str = None):
    folder_path = os.path.basename(tree_path)
    print(folder_path)
    temp_folder = os.path.join("tmp", folder_path)
    print(temp_folder)
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.mkdir(temp_folder)
    shutil.copytree(tree_path,temp_folder,dirs_exist_ok=True)
    return temp_folder