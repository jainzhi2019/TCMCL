import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MSA(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super(MSA, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (self.dim // self.heads) ** -0.5
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):  # b 50 768
        q = self.wq(x)  # b 50 768
        k = self.wk(x)  # b 50 768
        v = self.wv(x)  # b 50 768
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) ->b h n d', h=self.heads), [q, k, v])
        dots = (q @ k.transpose(-2, -1)) * self.scale  # b h n n
        attention = dots.softmax(dim=-1)
        out = torch.matmul(attention, v)
        out = rearrange(out, 'b h n d ->b n (h d)')
        return self.to_out(out)

class MCA(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super(MCA, self).__init__()
        self.dim = dim #50
        self.heads = heads
        self.scale = (self.dim // self.heads) ** -0.5
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x1, x2=None):  # b 50 768
        q = self.wq(x1)  # b 50 768
        k = self.wk(x2)  # b 50 47
        v = self.wv(x2)  # b 50 47
        #多头拆分
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) ->b h n d', h=self.heads), [q, k, v])
        #捕获维度间（748*47）的关系
        dots = (q @ k.transpose(-2, -1)) * self.scale  # b h n n
        attention = dots.softmax(dim=-1)
        #用音频强化文本特征
        out = torch.matmul(attention, v)
        #748*50  文本每个维度更关注音频中的哪个token（哪一帧）
        out = rearrange(out, 'b h n d ->b n (h d)')
        # return self.to_out(out)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),  #hidden_dim 200
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x1, x2=None, **kwargs):
        if x2 is not None:
            return self.fn(self.norm(x1), self.norm(x2), **kwargs)
        return self.fn(self.norm(x1), **kwargs)

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList(
                [PreNorm(dim, MSA(dim, heads=heads, dropout=dropout)),
                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout))]
            ))

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TransformerDecoder(nn.Module):
    #维度，深度（crossattention的个数），多头头数，mlp中间那层的特征维度
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList(
                [PreNorm(dim, MCA(dim, heads=heads, dropout=dropout)),
                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout))]
            ))

    def forward(self, x1, x2):

        for attn, ff in self.layers:
            x1 = attn(x1, x2) + x1
            x1 = ff(x1) + x1
        return x1


class MFG(nn.Module):
    def __init__(self, visual_dim, acoustic_dim, text_dim):
        super(MFG, self).__init__()
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, text_dim),
            nn.GELU(),
            # nn.TransformerEncoderLayer(text_dim, 12, dim_feedforward=text_dim * 4, dropout=0.5, activation=F.gelu,
            #                            batch_first=True)
        )
        self.acoustic_proj = nn.Sequential(
            nn.LayerNorm(acoustic_dim),
            nn.Linear(acoustic_dim, text_dim),
            nn.GELU(),
            # nn.TransformerEncoderLayer(text_dim, 12, dim_feedforward=text_dim * 4, dropout=0.5, activation=F.gelu,
            #                            batch_first=True)
        )

    def forward(self, visual, acoustic):
        visual_embedding = self.visual_proj(visual)
        acoustic_embedding = self.acoustic_proj(acoustic)

        return visual_embedding, acoustic_embedding


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.linear=nn.Linear(50 * 74, 1)

    def forward(self, input_ids,
                visual,
                acoustic,
                token_type_ids,
                attention_mask,
                labels=None, ):
        return self.linear(acoustic.view(acoustic.size(0), -1)),visual
