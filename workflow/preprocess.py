from . import node

# 预处理，虽然我也不知道干嘛用


class PreProcess(node.Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)

    def process(self):
        return super().process()
