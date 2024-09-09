#用于调整亮度
from . import node


class Exposure(node.Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
    def process(self):
        super().process()