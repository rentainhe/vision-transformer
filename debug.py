import math
import torch
from torch import nn
import numpy as np
from models.vit import Transformer
from models.performer import Performer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def exists(val):
    return val is not None


def conv_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)


# classes

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(x.shape[1])))


class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """

    def __init__(self, image_size=224, tokens_type='performer', in_chans=3, embed_dim=768, dropout=0.1,
                 t2t_layers=((7, 4), (3, 2), (3, 2))):
        super().__init__()
        layers = []
        layer_dim = in_chans
        output_image_size = image_size

        for i, (kernel_size, stride) in enumerate(t2t_layers):
            layer_dim *= kernel_size ** 2
            is_first = i == 0
            output_image_size = conv_output_size(output_image_size, kernel_size, stride, stride // 2)

            layers.extend([
                RearrangeImage() if not is_first else nn.Identity(),
                nn.Unfold(kernel_size=kernel_size, stride=stride, padding=stride // 2),
                Rearrange('b c n -> b n c'),
                Transformer(dim=layer_dim, heads=1, depth=1, dim_head=layer_dim, mlp_dim=layer_dim,
                            dropout=dropout) if tokens_type == 'transformer'
                else Performer(dim=layer_dim, inner_dim=layer_dim, kernel_ratio=0.5),
            ])
        layers.append(nn.Linear(layer_dim, embed_dim))
        self.to_patch_embedding = nn.Sequential(*layers)

    def forward(self, image):
        x = self.to_patch_embedding(image)
        return x