import os
import shutil
from img_class import TextureImage as timg
from BuildingObj import BuildingObj
from typing import *
from datetime import datetime
import cv2
temp_folder = "./tmp"

# def load_dta(input_path: str = None, output_path: str = None):
#     arg_list = []
#     script_path = os.getcwd()
#     # print(script_path)
#     if input_path is None:
#         raise ValueError("input_path is None")
#     if output_path is None:
#         raise ValueError("output_path is None")
#     if not os.path.isabs(input_path):
#         input_path = os.path.abspath(input_path)
#     if not os.path.isabs(output_path):
#         output_path = os.path.abspath(output_path)
#     print(input_path, output_path)
#     if os.path.isfile(input_path):
#         if input_path.endswith(".obj"):
#             print("input path is a single file")
#             output_folder = create_output_folder(os.path.dirname(input_path), output_path)
#             # print(output_folder)
#             print(f"Loading images from {input_path}")
#             with open(input_path, "r") as f:
#                 lines = f.readlines()
#                 for line in lines:
#                     if line.startswith("mtllib"):
#                         os.chdir(os.path.dirname(input_path))
#                         mtl_path = line.split(" ")[1]
#                         mtl_path = mtl_path.replace("\n","")
#                         mtl_handel(mtl_path)
#                         shutil.copy(mtl_path, os.path.join(output_folder, mtl_path))
#                         shutil.copy(input_path, os.path.join(output_folder, os.path.basename(os.path.basename(input_path))))
#                         mtl_path = os.path.abspath(mtl_path)
#                         os.chdir(script_path)
#                         temp_path = create_temp_folder(output_folder)
#                         temp_path = os.path.abspath(temp_path)
#                         arg_list.append((input_path, mtl_path, temp_path,output_folder))
#                         break
#     elif os.path.isdir(input_path):
#             #当路径是文件夹时，处理所有obj
#             print("input_path is a folder")
#             for root, dirs, files in os.walk(input_path):
#                 # input_folder = root     # 按理来说这里应当是绝对路径
#                 if len(files) == 0:
#                     continue
#                 for file in files:
#                     if file.endswith(".obj"):
#                         output_folder = create_output_folder(root, output_path)
#                         print(f"output folder{output_folder}")
#                         obj_path = os.path.join(root, file)
#                         with open(obj_path, "r") as f:
#                             while True:
#                                 os.chdir(root)
#                                 content = f.readline()
#                                 if content.startswith("mtllib"):
#                                     mtl_path = content.split(" ")[1]
#                                     mtl_path = os.path.join(os.path.dirname(obj_path), mtl_path)
#                                     # 我不知道为什么这里会多一个换行符
#                                     mtl_path = mtl_path.replace("\n", "")
#                                     mtl_handel(mtl_path)
#                                     print("mtl_path:", mtl_path)
#                                     shutil.copy(mtl_path, os.path.join(output_folder, os.path.basename(mtl_path)))
#                                     shutil.copy(obj_path, os.path.join(output_folder, os.path.basename(input_path)))
#                                     mtl_path = os.path.abspath(mtl_path)
#                                     os.chdir(script_path)
#                                     temp_path = create_temp_folder(output_folder)
#                                     temp_path = os.path.abspath(temp_path)
#                                     arg_list.append((input_path, mtl_path, temp_path,output_path))
#                                     break
#                         break
#     else:
#         raise ValueError("input_path is not an obj file")
#     return load_building(arg_list)

def is_filename(input_path:str = None):
    if os.path.exists(input_path):
        if os.path.isfile(input_path):
            return True
        else:
            return False
    else:
        dirname, filename = os.path.split(input_path)
        if '.' in filename:
            return True
        else:
            return False
def load_building(arg_list):
    for obj_path,mtl_path,temp_path,output_path in arg_list:
        obj_path,mtl_path,temp_path,output_path = normalise_path(obj_path,mtl_path,temp_path,output_path)
        yield BuildingObj(obj_path,mtl_path,temp_path,output_path)
def is_image_file(file_path: str = None):
    try:
        img = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)
        if img is not None:
            return True
    except Exception as e:
        print(e)
        return False


def mtl_handel(mtl_path: str =None):
    mtl_path = os.path.abspath(mtl_path)
    print(mtl_path)
    dir_name = os.path.dirname(mtl_path)
    script_path = os.getcwd()
    os.chdir(dir_name)
    new_file = open("new.tmp","a+",encoding='utf-8')
    with open(mtl_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.find("map_Kd") != -1:
                img_route = line.split(" ", 1)[-1]
                if os.path.isabs(img_route):
                    if not os.path.exists(img_route):
                        # 如果绝对路径不存在，尝试使用相对路径
                        img_route = os.path.basename(img_route)
                    else:
                        # 将绝对路径复制到同文件夹下
                        shutil.copy2(img_route)
                    # 改用相对路径
                    line = "map_Kd"+" "+img_route
            new_file.write(line)
    new_file.close()
    if os.path.exists("backup_mtl.txt"):
        os.remove("backup_mtl.txt")
    os.rename(os.path.basename(mtl_path),"backup_mtl.txt")
    os.rename("new.tmp",os.path.basename(mtl_path))
    os.chdir(script_path)
    return True

def create_output_folder(input_path:str = None,output_path: str = None):
    folder_path = os.path.dirname(input_path)
    output_folder = os.path.dirname(output_path)
    # print(output_folder)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder,exist_ok=True)
    shutil.copy2(input_path,output_path)
    with open(os.path.join(output_folder, "log.txt"), "w+") as f:
        f.write(f"Folder originate from {input_path}\n")
    return True
def create_temp_folder(input_path: str = None,temp_path:str = None):
    dirname,basename = os.path.split(temp_path)
    if '.' in basename:
        temp_path = dirname
    # else:
    #     temp_path = temp_path
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    print(temp_path)
    os.makedirs(temp_path,exist_ok=True)
    shutil.copy2(input_path,temp_path)
    return True
def normalise_path(obj_path,mtl_path,temp_path,output_path):
    # 将代码中的路径转换为绝对路径
    obj_path  = os.path.normpath(obj_path)
    mtl_path = os.path.normpath(mtl_path)
    temp_path = os.path.normpath(temp_path)
    output_path = os.path.normpath(output_path)
    obj_path = os.path.abspath(obj_path)
    mtl_path = os.path.abspath(mtl_path)
    temp_path = os.path.abspath(temp_path)
    output_path = os.path.abspath(output_path)
    # 若tmp/output带有文件名，删除
    if is_filename(temp_path):
        temp_path = os.path.dirname(temp_path)
    if is_filename(output_path):
        output_path = os.path.dirname(output_path)
    return obj_path,mtl_path,temp_path,output_path
def pack_building_object(obj_path,mtl_path,temp_path,output_path):
    yield BuildingObj(obj_path, mtl_path, temp_path, output_path)

def collect_obj(input_path: str = None):
    obj_path_list = []
    has_subdirectories = False
    for dirpath, dirnames, filenames in os.walk(input_path):
        if dirnames:
            has_subdirectories = True
        for name in filenames:
            if name.endswith(".obj"):
                obj_path = os.path.join(dirpath, name)
                obj_path_list.append(obj_path)
    return obj_path_list, has_subdirectories

def load_data(input_path,output_path):
    if not os.path.exists(input_path):
        raise ValueError("input_path is not exist")
    if os.path.isdir(input_path):
        print("input path is a folder")
        # obj_path_list = []
        # for dirpath,dirnames,filenames in os.walk(input_path):
        #     for name in filenames:
        #         if name.endswith(".obj"):
        #             obj_path = os.path.join(dirpath,name)
        #             obj_path_list.append(obj_path)
        obj_path_list,has_sub_directories = collect_obj(input_path)
        print(obj_path_list)
            # obj_path = [name for name in filenames if name.endswith(".obj")]
            # obj_path = [os.path.join(dirpath,obj) for obj in obj_path]
            # obj_path_list.append(obj_path)
        if not has_sub_directories:
            for obj_path in obj_path_list:
                output_path = os.path.join(output_path,os.path.basename(input_path))
                yield from load_data(obj_path,output_path)
                return
        rel_input_path = [os.path.relpath(obj,input_path) for obj in obj_path_list]
        # 生成在输出文件夹下的树状结构
        output_path_list = [os.path.join(output_path,rel_path) for rel_path in rel_input_path]
        # 生成在缓存文件夹下的树状结构
        temp_path_list = [os.path.join(temp_folder,rel_path) for rel_path in rel_input_path]
        mtl_path_list = []
        for obj,output,temp in zip(obj_path_list,output_path_list,temp_path_list):
            folder = os.path.dirname(obj)
            with open(obj,"r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("mtllib"):
                        mtl_path = line.split(" ")[1]
                        mtl = os.path.join(folder, mtl_path)
                        mtl = mtl.strip()
                        break
            # 到底为什么会有换行符？？
            # mtl = mtl.replace("\n","")
            mtl_handel(mtl)
            create_output_folder(obj,output)
            create_temp_folder(obj,temp)
            # 复制mtl文件到输出文件夹和缓存文件夹

            shutil.copy2(mtl,os.path.dirname(output))
            shutil.copy2(mtl,os.path.dirname(temp))
            mtl_path_list.append(mtl)
            # 打包obj模型
            # pack_building_object(obj,mtl,temp,output)
        yield from load_building(zip(obj_path_list,mtl_path_list,temp_path_list,output_path_list))
        return
    elif os.path.isfile(input_path):
        if input_path.endswith('.obj'):
            print("input path is a obj")
            obj = input_path
            input_folder = os.path.dirname(obj)
            with open(obj, 'r') as file:
                for line in file:
                    if line.startswith("mtllib"):
                        mtl_path = line.split(" ")[1]
                        mtl = os.path.join(input_folder, mtl_path)
                        mtl = mtl.strip()
                        break
            print(f"obj_path is{obj}",f"mtl_path is{mtl}\n ",f"temp_path is{temp_folder}\n",f"output_path is{output_path}\n")
            output_path = os.path.join(output_path, os.path.basename(input_folder))
            temp_path = os.path.join(temp_folder, os.path.basename(input_folder))
            print(temp_path)
            mtl_handel(mtl)
            create_output_folder(obj, output_path)
            create_temp_folder(obj, temp_path)
            shutil.copy2(mtl, output_path)
            shutil.copy2(mtl, temp_path)
            yield from load_building([(obj, mtl, temp_path, output_path)])
            return
        elif is_image_file(input_path):
            img_format = input_path.split(".")[-1]
            print("input path is an image")
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d")
            img_name = f"{now.strftime('%H%M%S')}.{img_format}"
            output_path = os.path.join(output_path, current_time)
            os.makedirs(output_path,exist_ok=True)
            temp_path = os.path.join(temp_folder, current_time)
            os.makedirs(temp_path,exist_ok=True)
            empty_building_object = BuildingObj(None, None, temp_path, output_path)
            texture_image = timg(input_path)
            texture_image.building_obj = empty_building_object
            texture_image.name = img_name
            empty_building_object.texture_list.append(texture_image)
            yield empty_building_object
            return
        else:
            raise(ValueError("input path does not "))