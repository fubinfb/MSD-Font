import torch
import torch.nn as nn
from functools import partial
import torch.nn as nn
from ldm.FFG_codebase.base.modules import ConvBlock, ResBlock

    
class StyleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        ConvBlk = partial(ConvBlock, norm="in", activ="relu", pad_type="zero")

        C = 32
        self.layers = nn.Sequential(
            ConvBlk(1, C, 3, 1, 1, norm='none', activ='none'),
            ConvBlk(C*1, C*2, 3, 1, 1, downsample=True),
            ConvBlk(C*2, C*2, 3, 1, 1, downsample=False),
            ConvBlk(C*2, C*4, 3, 1, 1, downsample=True),
            ConvBlk(C*4, C*4, 3, 1, 1, downsample=False)
        )

    def forward(self, x):
        style_feat = self.layers(x)
        return style_feat
    
class SingleExpert(nn.Module):
    def __init__(self):
        super().__init__()
        ResBlk = partial(ResBlock, norm="in", activ="relu", pad_type="zero", scale_var=False)

        C = 32
        self.layers = nn.ModuleList([
            ResBlk(C*4, C*4, 3, 1, downsample=True), 
            ResBlk(C*4, C*4, 3, 1),
        ])
        self.proj1 = nn.Sequential(
            ResBlk(C*4, C*4, 3, 1),
            ResBlk(C*4, C*4, 3, 1)
        )
        self.proj2 = nn.Sequential(
            ResBlk(C*4, C*4, 3, 1),
            ResBlk(C*4, C*4, 3, 1)
        )

    def forward(self, x):
        for lidx, layer in enumerate(self.layers):
            x = layer(x)
        char_code = self.proj1(x) 
        style_code = self.proj2(x) 

        return char_code, style_code

class miniUNet_enc_128(nn.Module):
    def __init__(self):
        super().__init__()
        self.style_enc = StyleEncoder()
        self.experts = SingleExpert()
        self.n_in_style = 3
    
    def encode(self, charimg):
        feats = self.style_enc(charimg)
        char_code, style_code = self.experts(feats) # 128, 16, 16

        return char_code, style_code