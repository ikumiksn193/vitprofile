import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

import time

# helpers
s_patch = 0
e_patch = 0
time_patch = []
s_pos = 0
e_pos = 0
time_pos = []
s_Tr = 0
e_Tr = 0
time_Tr = []
s_linear = 0
e_linear = 0
time_linear = []

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            # nn.GELU(),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        global s_patch 
        global e_patch 
        global time_patch 
        global s_pos 
        global e_pos 
        global time_pos 
        global s_Tr 
        global e_Tr 
        global time_Tr 
        global s_linear
        global e_linear
        global time_linear

        device = img.device
        s_patch = time.time()
        x = self.to_patch_embedding(img)
        e_patch = time.time()
        s_pos = time.time()
        x += self.pos_embedding.to(device, dtype=x.dtype)
        e_pos = time.time()
        s_Tr = time.time()
        x = self.transformer(x)
        e_Tr = time.time()
        x = x.mean(dim = 1)
        x = self.to_latent(x)
        s_linear = time.time()
        x = self.linear_head(x)
        e_linear = time.time()
        time_patch.append(e_patch - s_patch)
        time_pos.append(e_pos - s_pos)
        time_Tr.append(e_Tr - s_Tr)
        time_linear.append(e_linear - s_linear)
        return x

model = SimpleViT(
    image_size = 224,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)

# a = model.state_dict()
# b = list(a)
# sum = 0
# for name in b:
#     print(f"name : {name}, num of params : {a[name].numel()}")
#     sum += a[name].numel()
# print(sum)

inpt = torch.randn(1,3,224,224)

for i in range(100):
    model(inpt)
    
with open('data/patch.txt', mode='w') as f:
    for value in time_patch:
        f.write(str(value)+'\n')
        
with open('data/pos.txt', mode='w') as f:
    for value in time_pos:
        f.write(str(value)+'\n')
        
with open('data/Tr.txt', mode='w') as f:
    for value in time_Tr:
        f.write(str(value)+'\n')
        
with open('data/linear.txt', mode='w') as f:
    for value in time_linear:
        f.write(str(value)+'\n')
        