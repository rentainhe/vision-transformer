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
        self.output_image_size = output_image_size
        self.to_patch_embedding = nn.Sequential(*layers)

    def forward(self, image):
        x = self.to_patch_embedding(image)
        return x


# main class

class T2T_ViT(nn.Module):
    def __init__(self, *, image_size, num_classes, dim, depth=None, heads=None, mlp_dim=None, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., tokens_type='performer', transformer=None,
                 t2t_layers=((7, 4), (3, 2), (3, 2))):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.t2t_module = T2T_module(image_size=image_size, tokens_type=tokens_type, in_chans=channels, embed_dim=dim,
                                     dropout=dropout, t2t_layers=t2t_layers)
        # add position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.t2t_module.output_image_size ** 2 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if not exists(transformer):
            assert all([exists(depth), exists(heads), exists(mlp_dim)]), 'depth, heads, and mlp_dim must be supplied'
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.t2t_module(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == "__main__":
    v = T2T_ViT(
        dim=512,
        image_size=224,
        depth=5,
        heads=8,
        mlp_dim=512,
        num_classes=1000,
        t2t_layers=((7, 4), (3, 2), (3, 2))
        # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
    )

    img = torch.randn(1, 3, 224, 224)

    preds = v(img)  # (1, 1000)
    print(preds.size())
