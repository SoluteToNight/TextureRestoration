from .preprocess import PreProcess
from .analyse import Analyse
from .deshadow import Deshadow
from .diffusion import Diffusion
from .exposure import Exposure
from .masking import Masking



def preprocess(inputs: [list[str], str]):
    return PreProcess(inputs)


# 输入路径或路径列表
def analyse(inputs:[list[str], str]):
    return Analyse(inputs)


def deshadow(inputs):
    return Deshadow(inputs)


def diffusion(inputs):
    return Diffusion(inputs)


def exposure(inputs):
    return Exposure(inputs)


def masking(inputs):
    return Masking(inputs)
