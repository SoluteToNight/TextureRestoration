#用于调整亮度
from .node import Node


class Exposure(Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
    def process(self):
        super().process()