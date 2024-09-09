from . import node
#去阴影
class Deshadow(node.Node):
    def __init__(self,inputs=None):
        super().__init__(inputs)
    def process(self):
        super().process()