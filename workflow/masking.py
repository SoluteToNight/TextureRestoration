from . import node
#蒙版
class Masking(node.Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
    def process(self):
        super().process()