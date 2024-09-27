from .node import Node
#去阴影
class Deshadow(Node):
    def __init__(self,inputs=None):
        super().__init__(inputs)
    def process(self):
        super().process()