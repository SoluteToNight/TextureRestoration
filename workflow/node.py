class Node:
    def __init__(self, inputs=None):
        self.input = inputs
        self.output = None
        self.model = None  # 应用的模型
        # self.model = 加载模型

    def process(self):
        if self.input is not None:
            print(f"Get input {self.input}")
        # 运用模型进行处理
        self.output = self.input
        return self.output
