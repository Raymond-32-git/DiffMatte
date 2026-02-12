import torch.nn as nn
from detectron2.layers import ShapeSpec

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
    def output_shape(self):
        return {}
