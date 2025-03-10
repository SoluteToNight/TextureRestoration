import os
import shutil
from img_class import TextureImage as timg
from BuildingObj import BuildingObj
from typing import *

temp_folder = "./tmp"

def load_dta(input_path: str = None, output_path: str = None):
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
                        arg_list.append((input_path, mtl_path, temp_path,output_folder))
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
                                    arg_list.append((input_path, mtl_path, temp_path,output_path))
                                    break
                        break
    else:
        raise ValueError("input_path is not an obj file")
    return load_building(arg_list)
def load_building(arg_list):
    for obj_path,mtl_path,temp_path,output_path in arg_list:
        yield BuildingObj(obj_path,mtl_path,temp_path,output_path)

def mtl_handel(mtl_path: str =None):
    print(mtl_path)
    dir_name = os.path.dirname(mtl_path)
    img_path = []
    with open(mtl_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.find("map_Kd") != -1:
                img_route = line.split(" ", 1)
                if os.path.isabs(img_route):
                    if not os.path.exists(img_route):
                        # 如果绝对路径不存在，尝试使用相对路径
                        img_route = os.path.basename(img_route)
                img_path.append(img_route)
    return img_path
def create_output_folder(input_path:str = None,output_path: str = None):
    folder_path = os.path.dirname(input_path)
    output_folder = os.path.dirname(output_path)
    print(output_folder)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    shutil.copy2(input_path,output_path)
    with open(os.path.join(output_folder, "log.txt"), "w+") as f:
        f.write(f"Folder originate from {input_path}\n")
    return True
def create_temp_folder(input_path: str = None,temp_path:str = None):
    folder_path = os.path.dirname(input_path)
    tmp_folder = os.path.dirname(temp_path)
    print(tmp_folder)
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)
    shutil.copy2(input_path,temp_path)
    return True

def pack_building_object(obj_path,mtl_path,temp_path,output_path):
    print(f"Loading image from{obj_path}")
    folder = os.path.dirname(obj_path)
    with open(obj_path,'w') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("mtllib"):
                mtl_path = line.split(" ")[1]
                mtl_path = os.path.join(folder,mtl_path)
        print(mtl_path)
        # 根据mtl搜索图片
    img_path_list = mtl_handel(mtl_path)


def load_data(input_path,output_path):
    if os.path.isdir(input_path):
        print("input path is a folder")
        obj_path_list = []
        for dirpath,dirnames,filenames in os.walk(input_path):
            for name in filenames:
                if name.endswith(".obj"):
                    obj_path = os.path.join(dirpath,name)
                    obj_path_list.append(obj_path)

            # obj_path = [name for name in filenames if name.endswith(".obj")]
            # obj_path = [os.path.join(dirpath,obj) for obj in obj_path]
            # obj_path_list.append(obj_path)
        rel_input_path = [os.path.relpath(obj,input_path) for obj in obj_path_list]
        # 生成在输出文件夹下的树状结构
        output_path_list = [os.path.join(output_path,rel_path) for rel_path in rel_input_path]
        # 生成在缓存文件夹下的树状结构
        temp_path_list = [os.path.join(temp_folder,rel_path) for rel_path in rel_input_path]
        for obj,output,temp in zip(obj_path_list,output_path_list,temp_path_list):
            folder = os.path.dirname(obj)
            with open(obj,"r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("mtllib"):
                        mtl_path = line.split(" ")[1]
                        mtl = os.path.join(folder, mtl_path)
            create_output_folder(obj,output)
            create_temp_folder(obj,temp)
            shutil.copy2(mtl,os.path.dirname(output))
            shutil.copy2(mtl,os.path.dirname(temp))
            pack_building_object(obj,mtl,temp,output)
    elif os.path.isfile(input_path):
        if input_path.endswith('.obj'):
            print("input path is a obj")
        elif input_path.endswith('.png', 'jpg'):
            print("input path is an image")