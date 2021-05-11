import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.transformer import Transformer

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# # classes
# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn
#
#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)
#
#
# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# class Attention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         # When heads > 1 it needs to be linear merged
#         # When dim_head != dim, it should change the output dim
#         project_out = not (heads == 1 and dim_head == dim)
#
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#
#         self.attend = nn.Softmax(dim=-1)
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
#
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()
#
#     def forward(self, x):
#         # batch, token_nums, channel, heads
#         b, n, _, h = *x.shape, self.heads
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
#         # qk -> dots product
#         dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
#         # attention map
#         attn = self.attend(dots)
#         # attend @ v
#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)
#
#
# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
#             ]))
#
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        """
            image_size: 输入的图片大小
            patch_size: 分块大小
            num_classes: 分类数目
            dim: Transformer Encoder输出的维度
            depth: 层数
            dropout: dropout rate in FFN
            emb_dropout: dropout rate in Patch Projection
            pool: cls token pooling or mean pooling
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        # patch embedding dropout
        x = self.dropout(x)

        x = self.transformer(x)
        # use cls token for classification or mean pool feature
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
