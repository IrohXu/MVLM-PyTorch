import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import PatchEmbed, Mlp, DropPath

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        B1, N1, C1 = x1.shape
        B2, N2, C2 = x2.shape
        qkv1 = self.qkv1(x1).reshape(B1, N1, 3, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv2(x2).reshape(B2, N2, 3, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q2, k2, v2 = qkv2.unbind(0) 
                
        attn11 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn11 = attn11.softmax(dim=-1)
        attn11 = self.attn_drop(attn11)
        
        attn12 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn12 = attn12.softmax(dim=-1)
        attn12 = self.attn_drop(attn12)
        
        attn21 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn21 = attn21.softmax(dim=-1)
        attn21 = self.attn_drop(attn21)
        
        attn22 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn22 = attn22.softmax(dim=-1)
        attn22 = self.attn_drop(attn22)

        x1 = (attn11 @ v1).transpose(1, 2).reshape(B1, N1, C1) + (attn12 @ v2).transpose(1, 2).reshape(B1, N1, C1)
        x2 = (attn22 @ v2).transpose(1, 2).reshape(B2, N2, C2) + (attn21 @ v1).transpose(1, 2).reshape(B2, N2, C2)
        x1 = self.proj1(x1)
        x1 = self.proj_drop(x1)
        x2 = self.proj2(x2)
        x2 = self.proj_drop(x2)
        return x1, x2

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class CrossBlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm11 = norm_layer(dim)
        self.norm12 = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls11 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls12 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm21 = norm_layer(dim)
        self.norm22 = norm_layer(dim)
        self.mlp1 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls21 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls22 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x1, x2):
        
        x1 = self.norm11(x1)
        x2 = self.norm12(x2)
        x1, x2 = self.attn(x1, x2)
        x1 = self.ls11(x1)
        x2 = self.ls12(x2)
        x1 = x1 + self.drop_path1(x1)
        x2 = x2 + self.drop_path1(x2)
        x1 = x1 + self.drop_path2(self.ls21(self.mlp1(self.norm21(x1))))
        x2 = x2 + self.drop_path2(self.ls22(self.mlp2(self.norm22(x2))))
        return x1, x2


# model = Block(dim=768, num_heads=8)
# x1 = torch.randn((2, 196, 768))
# x2 = torch.randn((2, 256, 768))
# print(x1.shape, x2.shape)
# print(model)
# x1, x2 = model(x1, x2)
# print(x1.shape, x2.shape)