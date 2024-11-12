import os

from img_class import TextureImage


class BuildingObj:
    def __init__(self, obj_path, mtl_path, temp_path,output_path):
        self.obj_path = obj_path
        self.mtl_path = mtl_path
        self.texture_list: list[TextureImage] = []
        self.temp_path = temp_path
        self.output_path = output_path
        print("Loading image from:", obj_path)
        print(f"obj path{obj_path}")
        print(f"mtl path{mtl_path}")
        print(f"temp path{temp_path}")
        print(f"output path{output_path}")
    def load_texture(self):
        script_path = os.getcwd()
        print(self.mtl_path)
        os.chdir(os.path.dirname(self.mtl_path))
        print(os.getcwd())
        with open(self.mtl_path) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith("map_Kd"):
                    texture_path = line.split(" ")[1]
                    texture_path = os.path.abspath(texture_path)
                    img = TextureImage(texture_path)
                    img.building_obj = self
                    self.texture_list.append(img)

            os.chdir(script_path)
        print( len(self.texture_list),"were loaded")