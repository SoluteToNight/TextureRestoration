from . import node
from img_class import TextureImage as timg
class Diffusion(node.Node):
    def __init__(self,inputs=None):
        super().__init__(inputs)
        self.model = "./models/architecturerealmix"
    def process(self):
        return