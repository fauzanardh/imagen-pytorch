import numpy as np
from functools import partial
from einops import rearrange, pack, unpack

import torch
from torch import nn

from imagen_pytorch.imagen_pytorch import (
    ResnetBlock,
    Downsample,
    CrossEmbedLayer,
    Upsample,
    PixelShuffleUpsample,
    FlashAttention,
    Attention,
)


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, min=-30, max=20)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        return self.mean + self.std * torch.randn_like(
            self.mean, device=self.parameters.device
        )

    def kl(self, other=None):
        if other is None:
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1 - self.logvar, dim=[1, 2, 3]
            )
        else:
            return 0.5 * torch.sum(
                torch.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                dim=[1, 2, 3],
            )

    def nll(self, sample, dims=[1, 2, 3]):
        log2pi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            log2pi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class Encoder(nn.Module):
    def __init__(
        self,
        in_dim,
        dim_mults=(1, 2, 4, 8),
        resolution=512,
        z_channels=4,
        double_z=True,
        in_channels=3,
        num_resnet_blocks=2,
        resnet_groups=8,
        attn_enabled=(False, True, True, True),
        attn_mid_enabled=False,
        attn_heads=8,
        attn_dim_head=64,
        cosine_sim_attn=False,
        cross_embed_downsample=False,
        cross_embed_downsample_kernel_sizes=(2, 4),
        use_flash_attn=True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_resolutions = len(dim_mults)
        self.num_resnet_blocks = num_resnet_blocks

        self.init_conv = nn.Conv2d(
            in_channels,
            in_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        attn_kwargs = dict(
            heads=attn_heads,
            dim_head=attn_dim_head,
            cosine_sim_attn=cosine_sim_attn,
        )
        resnet_klass = partial(
            ResnetBlock, use_flash_attn=use_flash_attn, **attn_kwargs
        )
        attn_klass = Attention if not use_flash_attn else FlashAttention

        downsample_klass = Downsample
        if cross_embed_downsample:
            downsample_klass = partial(
                CrossEmbedLayer,
                kernel_sizes=cross_embed_downsample_kernel_sizes,
            )

        current_res = resolution
        in_dim_mults = (1,) + tuple(dim_mults)
        self.in_dim_mults = in_dim_mults
        self.down = nn.ModuleList()
        for i in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = in_dim * in_dim_mults[i]
            block_out = in_dim * dim_mults[i]
            for _ in range(num_resnet_blocks):
                block.append(
                    resnet_klass(
                        block_in,
                        block_out,
                        groups=resnet_groups,
                    )
                )
                if attn_enabled[i]:
                    attn.append(
                        attn_klass(
                            block_out,
                            **attn_kwargs,
                        )
                    )
                else:
                    attn.append(None)
                block_in = block_out

            downsample = None
            if i != self.num_resolutions - 1:
                downsample = downsample_klass(block_in)
                current_res //= 2

            down = nn.Module()
            down.block = block
            down.attn = attn
            down.downsample = downsample
            self.down.append(down)

        self.mid = nn.ModuleList()
        for _ in range(num_resnet_blocks):
            block = resnet_klass(
                block_in,
                block_in,
                groups=resnet_groups,
            )
            attn = None
            if attn_mid_enabled:
                attn = attn_klass(
                    block_in,
                    **attn_kwargs,
                )
            mid = nn.Module()
            mid.block = block
            mid.attn = attn
            self.mid.append(mid)

        self.norm_out = nn.GroupNorm(32, block_in)
        self.gelu = nn.GELU()
        self.conv_out = nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        x = self.init_conv(x)
        for i in range(self.num_resolutions):
            for resnet, attn in zip(self.down[i].block, self.down[i].attn):
                x = resnet(x)
                if attn is not None:
                    x = rearrange(x, "b c h w -> b h w c")
                    x, ps = pack([x], "b * c")
                    x = attn(x) + x
                    (x,) = unpack(x, ps, "b * c")
                    x = rearrange(x, "b h w c -> b c h w")
            if self.down[i].downsample is not None:
                x = self.down[i].downsample(x)

        for layer in self.mid:
            x = layer.block(x)
            if layer.attn is not None:
                x = rearrange(x, "b c h w -> b h w c")
                x, ps = pack([x], "b * c")
                x = layer.attn(x) + x
                (x,) = unpack(x, ps, "b * c")
                x = rearrange(x, "b h w c -> b c h w")

        x = self.norm_out(x)
        x = self.gelu(x)
        x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_dim,
        dim_mults=(1, 2, 4, 8),
        resolution=512,
        z_channels=4,
        in_channels=3,
        num_resnet_blocks=2,
        resnet_groups=8,
        attn_enabled=(False, True, True, True),
        attn_mid_enabled=False,
        attn_heads=8,
        attn_dim_head=64,
        cosine_sim_attn=False,
        pixel_shuffle_upsample=False,
        use_flash_attn=True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_resolutions = len(dim_mults)
        self.num_resnet_blocks = num_resnet_blocks

        attn_kwargs = dict(
            heads=attn_heads,
            dim_head=attn_dim_head,
            cosine_sim_attn=cosine_sim_attn,
        )
        resnet_klass = partial(
            ResnetBlock, use_flash_attn=use_flash_attn, **attn_kwargs
        )
        attn_klass = Attention if not use_flash_attn else FlashAttention

        upsample_klass = (
            Upsample if not pixel_shuffle_upsample else PixelShuffleUpsample
        )

        in_dim_mults = (1,) + tuple(dim_mults)
        block_in = in_dim * in_dim_mults[self.num_resolutions - 1]
        current_res = resolution // 2 ** (self.num_resolutions - 1)

        self.init_conv = nn.Conv2d(
            z_channels,
            block_in,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.mid = nn.ModuleList()
        for _ in range(num_resnet_blocks):
            block = resnet_klass(
                block_in,
                block_in,
                groups=resnet_groups,
            )
            attn = None
            if attn_mid_enabled:
                attn = attn_klass(
                    block_in,
                    **attn_kwargs,
                )
            mid = nn.Module()
            mid.block = block
            mid.attn = attn
            self.mid.append(mid)

        self.up = nn.ModuleList()
        for i in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = in_dim * dim_mults[i]
            for _ in range(num_resnet_blocks + 1):
                block.append(
                    resnet_klass(
                        block_in,
                        block_out,
                        groups=resnet_groups,
                    )
                )
                if attn_enabled[i]:
                    attn.append(
                        attn_klass(
                            block_out,
                            **attn_kwargs,
                        )
                    )
                else:
                    attn.append(None)
                block_in = block_out

            upsample = None
            if i != 0:
                upsample = upsample_klass(block_in)
                current_res *= 2

            up = nn.Module()
            up.block = block
            up.attn = attn
            up.upsample = upsample

            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(32, block_in)
        self.gelu = nn.GELU()
        self.conv_out = nn.Conv2d(
            block_in,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, z):
        z = self.init_conv(z)
        for layer in self.mid:
            z = layer.block(z)
            if layer.attn is not None:
                z = rearrange(z, "b c h w -> b h w c")
                z, ps = pack([z], "b * c")
                z = layer.attn(z) + z
                (z,) = unpack(z, ps, "b * c")
                z = rearrange(z, "b h w c -> b c h w")

        for i in reversed(range(self.num_resolutions)):
            for (resnet, attn) in zip(self.up[i].block, self.up[i].attn):
                z = resnet(z)
                if attn is not None:
                    z = rearrange(z, "b c h w -> b h w c")
                    z, ps = pack([z], "b * c")
                    z = attn(z) + z
                    (z,) = unpack(z, ps, "b * c")
                    z = rearrange(z, "b h w c -> b c h w")
            if self.up[i].upsample is not None:
                z = self.up[i].upsample(z)

        z = self.norm_out(z)
        z = self.gelu(z)
        z = self.conv_out(z)
        return z


class AutoEncoderKL(nn.Module):
    def __init__(
        self,
        in_dim,
        dim_mults=(1, 2, 4, 8),
        embed_dim=4,
        resolution=512,
        z_channels=4,
        double_z=True,
        in_channels=3,
        num_resnet_blocks=2,
        resnet_groups=8,
        attn_enabled=(False, True, True, True),
        attn_mid_enabled=False,
        attn_heads=8,
        attn_dim_head=64,
        cosine_sim_attn=False,
        cross_embed_downsample=False,
        cross_embed_downsample_kernel_sizes=(2, 4),
        pixel_shuffle_upsample=False,
        use_flash_attn=True,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_dim,
            dim_mults=dim_mults,
            resolution=resolution,
            z_channels=z_channels,
            double_z=double_z,
            in_channels=in_channels,
            num_resnet_blocks=num_resnet_blocks,
            resnet_groups=resnet_groups,
            attn_enabled=attn_enabled,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            cosine_sim_attn=cosine_sim_attn,
            cross_embed_downsample=cross_embed_downsample,
            cross_embed_downsample_kernel_sizes=cross_embed_downsample_kernel_sizes,
            use_flash_attn=use_flash_attn,
        )
        self.decoder = Decoder(
            in_dim,
            dim_mults=dim_mults,
            resolution=resolution,
            z_channels=z_channels,
            in_channels=in_channels,
            num_resnet_blocks=num_resnet_blocks,
            resnet_groups=resnet_groups,
            attn_enabled=attn_enabled,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            cosine_sim_attn=cosine_sim_attn,
            pixel_shuffle_upsample=pixel_shuffle_upsample,
            use_flash_attn=use_flash_attn,
        )

        self.quantize_conv = nn.Conv2d(
            2 * z_channels if double_z else z_channels,
            2 * embed_dim,
            kernel_size=1,
        )
        self.post_quantize_conv = nn.Conv2d(
            embed_dim,
            z_channels if double_z else z_channels,
            kernel_size=1,
        )

    def encode(self, x):
        z = self.encoder(x)
        moments = self.quantize_conv(z)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quantize_conv(z)
        x = self.decoder(z)
        return x

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        dec = self.decode(z)
        return dec, posterior
