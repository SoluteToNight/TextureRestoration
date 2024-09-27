from .node import Node
#蒙版
class Masking(Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
    def process(self):
        super().process()