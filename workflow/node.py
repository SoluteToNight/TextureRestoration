from img_class import TextureImage as timg


class Node:
    def __init__(self, inputs=None):
        self.img_list: list[timg] = inputs
        self.model = None  # 应用的模型
        # self.model = 加载模型

    def convert(self):
        # 对数据根据当前节点需要格式转换
        return

    def process(self):
        return

    def convert_back(self):
        return
