from .node import Node


class CCSR(Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
        self.model = None
    def process(self):
        return
