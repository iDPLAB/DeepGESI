# Copyright (c) the Lab of Intelligent Data Processing, Wakayama University.
# All rights reserved.

# Gesinet_module.py
import math
import torch
import torch.nn as nn
import torchaudio
from sincnet import SincConv_fast

sample_rate = 16000


# =========================================================
# Activation
# =========================================================
def build_activation(name: str):
    name = name.lower()

    if name == "relu":
        return nn.ReLU()

    elif name == "prelu":
        return nn.PReLU()

    elif name == "lrelu":
        return nn.LeakyReLU(negative_slope=0.1)

    elif name == "silu":
        return nn.SiLU()

    else:
        raise ValueError(f"Unknown activation: {name}")


# =========================================================
# Positional Encoding
# =========================================================
class SinusoidalAbsolutePE(nn.Module):
    def __init__(self, embed_dim: int = 128, max_len: int = 10000):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :].to(dtype=x.dtype, device=x.device)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even."
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        assert D % 2 == 0, "Last dimension must be even for RoPE."

        pos = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        sinusoid_inp = torch.einsum("i,j->ij", pos, self.inv_freq)

        sin = sinusoid_inp.sin()[None, None, :, :]
        cos = sinusoid_inp.cos()[None, None, :, :]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        x_rot = torch.stack(
            [x1 * cos - x2 * sin, x2 * cos + x1 * sin],
            dim=-1
        ).flatten(-2)

        return x_rot


class T5RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int = 8, max_distance: int = 128, per_head: bool = True):
        super().__init__()

        self.num_heads = num_heads
        self.max_distance = max_distance
        self.per_head = per_head

        table_size = 2 * max_distance + 1

        if per_head:
            self.relative_bias = nn.Parameter(torch.zeros(num_heads, table_size))
        else:
            self.relative_bias = nn.Parameter(torch.zeros(1, table_size))

        nn.init.normal_(self.relative_bias, mean=0.0, std=0.02)

    def forward(self, T: int, device=None, dtype=None) -> torch.Tensor:
        pos = torch.arange(T, device=device)
        rel_pos = pos[:, None] - pos[None, :]
        rel_pos = rel_pos.clamp(-self.max_distance, self.max_distance)
        index = rel_pos + self.max_distance

        bias = self.relative_bias[:, index]

        if dtype is not None:
            bias = bias.to(dtype=dtype)

        return bias


# =========================================================
# Attention
# =========================================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        dropout=0.0,
        pe_type="rope",
        t5_max_distance=128,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.pe_type = pe_type.lower()

        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if self.pe_type == "sinusoidal":
            self.abs_pe = SinusoidalAbsolutePE(embed_dim=embed_dim)

        elif self.pe_type == "rope":
            self.rope = RotaryPositionEmbedding(self.head_dim)

        elif self.pe_type == "t5":
            self.t5_bias = T5RelativePositionBias(
                num_heads=num_heads,
                max_distance=t5_max_distance,
                per_head=True,
            )

        elif self.pe_type == "none":
            pass

        else:
            raise ValueError(f"Unknown pe_type: {pe_type}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape

        if self.pe_type == "sinusoidal":
            x = self.abs_pe(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.pe_type == "rope":
            q = self.rope(q)
            k = self.rope(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.pe_type == "t5":
            bias = self.t5_bias(T, device=x.device, dtype=scores.dtype)
            scores = scores + bias.unsqueeze(0)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        out = self.out_proj(out)

        return out, attn


# =========================================================
# MaxOut
# =========================================================
class MaxOut(nn.Module):
    def __init__(self, in_features, out_features, pool_size=2):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features * pool_size)
        self.out_features = out_features
        self.pool_size = pool_size

    def forward(self, x):
        shape = list(x.shape[:-1]) + [self.out_features, self.pool_size]
        return self.fc(x).view(*shape).max(-1)[0]


# =========================================================
# CNN Blocks
# =========================================================
class ConvMaxOutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=2):
        super().__init__()

        self.pool_size = pool_size
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels * pool_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels * pool_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels * pool_size, kernel_size=3, stride=(1, 3), padding=1)

    def maxout(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.out_channels, self.pool_size, H, W)
        return x.max(dim=2)[0]

    def forward(self, x):
        x = self.maxout(self.conv1(x))
        x = self.maxout(self.conv2(x))
        x = self.maxout(self.conv3(x))
        return x


class ConvActivationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation="relu"):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = build_activation(activation)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = build_activation(activation)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=(1, 3), padding=1)
        self.act3 = build_activation(activation)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        return x


def build_conv_block(in_channels, out_channels, cnn_activation="maxout"):
    cnn_activation = cnn_activation.lower()

    if cnn_activation == "maxout":
        return ConvMaxOutBlock(in_channels, out_channels)

    elif cnn_activation in ["relu", "prelu", "lrelu", "silu"]:
        return ConvActivationBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=cnn_activation,
        )

    else:
        raise ValueError(f"Unknown cnn_activation: {cnn_activation}")


# =========================================================
# MLP Regressor
# =========================================================
def build_regressor(mlp_activation="maxout"):
    mlp_activation = mlp_activation.lower()

    if mlp_activation == "maxout":
        return nn.Sequential(
            MaxOut(128, 64),
            MaxOut(64, 32),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    elif mlp_activation in ["relu", "prelu", "lrelu", "silu"]:
        return nn.Sequential(
            nn.Linear(128, 64),
            build_activation(mlp_activation),
            nn.Linear(64, 32),
            build_activation(mlp_activation),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    else:
        raise ValueError(f"Unknown mlp_activation: {mlp_activation}")


# =========================================================
# Main Model
# =========================================================
class GESINet_Strict(nn.Module):
    def __init__(
        self,
        sinc_channels=257,
        sinc_kernel=512,
        embed_dim=128,
        num_heads=8,
        use_attention=True,
        pe_type="rope",
        cnn_activation="maxout",
        mlp_activation="maxout",
        attn_dropout=0.0,
        flatten_dropout=0.1,
        t5_max_distance=128,
    ):
        super().__init__()

        self.use_attention = use_attention
        self.pe_type = pe_type
        self.cnn_activation = cnn_activation
        self.mlp_activation = mlp_activation
        self.embed_dim = embed_dim

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=512,
            win_length=512,
            hop_length=256,
            power=2,
            window_fn=torch.hamming_window,
        )

        self.sincnet = SincConv_fast(
            out_channels=sinc_channels,
            kernel_size=sinc_kernel,
            sample_rate=sample_rate,
            stride=256,
        )

        self.proj_sinc = nn.Conv1d(sinc_channels, 257, kernel_size=1)

        self.conv1 = build_conv_block(1, 16, cnn_activation)
        self.conv2 = build_conv_block(16, 32, cnn_activation)
        self.conv3 = build_conv_block(32, 64, cnn_activation)
        self.conv4 = build_conv_block(64, 128, cnn_activation)

        self.flatten_fc = nn.Sequential(
            MaxOut(128 * 7, embed_dim),
            nn.Dropout(flatten_dropout),
        )

        if self.use_attention:
            self.attn = MultiHeadSelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                pe_type=pe_type,
                t5_max_distance=t5_max_distance,
            )
        else:
            self.attn = None

        self.regressor = build_regressor(mlp_activation=mlp_activation)

    def forward(self, x):
        """
        x: [B, 1, T]
        """
        B, _, T = x.shape

        stft_feat = self.stft(x.squeeze(1))
        stft_feat = torch.log1p(stft_feat + 1e-7).permute(0, 2, 1)

        sinc_feat = self.sincnet(x)
        sinc_feat = self.proj_sinc(sinc_feat).permute(0, 2, 1)

        T_min = min(stft_feat.shape[1], sinc_feat.shape[1])
        stft_feat = stft_feat[:, :T_min, :]
        sinc_feat = sinc_feat[:, :T_min, :]

        feat = torch.cat([stft_feat, sinc_feat], dim=2)  # [B, T, 514]
        feat = feat.unsqueeze(1)                         # [B, 1, T, 514]

        feat = self.conv1(feat)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv4(feat)

        B, C, H, W = feat.shape

        if W != 7:
            raise RuntimeError(
                f"Unexpected CNN output W={W}. "
                f"This model expects W=7 so that flatten input is 128*7."
            )

        feat = feat.permute(0, 2, 1, 3).reshape(B, H, C * W)
        feat = self.flatten_fc(feat)

        if self.use_attention:
            feat2, _ = self.attn(feat)
            feat = feat + feat2

        frame_scores = self.regressor(feat)
        sentence_score = frame_scores.mean(dim=1)

        return sentence_score, frame_scores
