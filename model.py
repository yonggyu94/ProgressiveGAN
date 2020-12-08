import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm


''' Device type'''
dev = 'cpu'
if torch.cuda.is_available():
    dev = 'cuda'


class WrongNormException(Exception):
    def __str__(self):
        return 'You should choose \'BN\', \'IN\' or \'SN\''


class WrongNonLinearException(Exception):
    def __str__(self):
        return 'You should choose \'relu\', \'leaky_relu\' or \'tanh\''


class AffineBlock(nn.Module):
    def __init__(self, in_dim, out_dim, n_slope=0.01, norm="SN", non_linear='relu'):
        super(AffineBlock, self).__init__()
        layers = []

        if norm == "SN":
            layers.append(spectral_norm(nn.Linear(in_dim, out_dim)))
        else:
            layers.append(nn.Linear(in_dim, out_dim))
            if norm == 'BN':
                layers.append(nn.BatchNorm1d(out_dim, affine=True))
            elif norm == 'IN':
                layers.append(nn.InstanceNorm1d(out_dim, affine=True))
            elif norm == None: pass
            else: raise WrongNormException()

        if non_linear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif non_linear == 'leaky_relu':
            layers.append(nn.LeakyReLU(n_slope, inplace=True))
        elif non_linear == 'sigmoid':
            layers.append(nn.Sigmoid(inplace=True))
        elif non_linear == 'tanh':
            layers.append(nn.Tanh(inplace=True))
        elif non_linear == None:
            pass
        else:
            raise WrongNonLinearException()

        self.affine_block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.affine_block(x)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ksize=3, stride=1, padding=1, n_slope=0.01, downsample=False,
                 norm='SN', non_linear='leaky_relu'):
        super(ConvBlock, self).__init__()
        layers = []

        if norm == 'SN':
            layers.append(spectral_norm(nn.Conv2d(in_dim, out_dim, kernel_size=ksize, stride=stride, padding=padding)))
        else:
            layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=ksize, stride=stride, padding=padding))

            if norm == 'BN':
                layers.append(nn.BatchNorm2d(out_dim, affine=True))
            elif norm == 'IN':
                layers.append(nn.InstanceNorm2d(out_dim, affien=True))
            elif norm == None : pass
            else: raise WrongNormException

        if downsample == True:
            layers.append(nn.AvgPool2d(2))

        if non_linear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif non_linear == 'leaky_relu':
            layers.append(nn.LeakyReLU(n_slope, inplace=True))
        elif non_linear == 'sigmoid':
            layers.append(nn.Sigmoid(inplace=True))
        elif non_linear == 'tanh':
            layers.append(nn.Tanh(inplace=True))
        elif non_linear == None: pass
        else: raise WrongNonLinearException

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_block(x)
        return out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, w_dim, in_dim):
        super(AdaptiveInstanceNorm, self).__init__()
        self.in_dim = in_dim
        self.linear = nn.Linear(w_dim, self.in_dim*2)
        self.instance_norm = nn.InstanceNorm2d(inplace=True)

    def forward(self, x, w):
        w_out = self.linear(w)
        out = self.instance_norm(x)         # [B, in_dim, H, W]

        style_mean, style_std = torch.split(w_out, self.in_dim, dim=1)      # [B, in_dim], [B, in_dim]
        out = out * style_mean.expand_as(out) + style_std.expand_as(out)    # [B, in_dim, H, W]
        return out


class NoiseInjection(nn.Module):
    def __init__(self, in_dim):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, in_dim, 1, 1))

    def forward(self, x, noise):
        out = x + self.weight * noise
        return out


class DownscaleBlock(nn.Module):
    def __init__(self, in_dim, out_dim, n_slope=0.2):
        super(DownsampleBlock, self).__init__()
        self.conv1 = ConvBlock(in_dim, in_dim, ksize=3, stride=1, padding=1, n_slope=n_slope, norm=None,
                               non_linear='leaky_relu')
        self.conv2 = ConvBlock(in_dim, out_dim, ksize=3, stride=1, padding=1, n_slope=n_slope, downsample=True,
                               norm=None, non_linear='leaky_relu')

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class SynthesisConstBlock(nn.Module):
    def __init__(self, w_dim, in_dim, out_dim):
        super(SynthesisConstBlock, self).__init__()
        self.noise_inject1 = NoiseInjection(in_dim=in_dim)
        self.adain1 = AdaptiveInstanceNorm(w_dim, in_dim)
        self.conv = ConvBlock(in_dim, out_dim, ksize=3, stride=1, padding=1, n_slope=0.2, norm=None,
                              non_linear='leaky_relu')
        self.noise_inject2 = NoiseInjection(in_dim=in_dim)
        self.adain2 = AdaptiveInstanceNorm(w_dim, in_dim)

    def forward(self, x, w, noise):
        noise1, noise2 = torch.split(noise, 2, dim=1)   # [B, 1, H, W], [B, 1, H, W]
        out = self.noise_inject1(x, noise1)     # [B, C, H, W]
        out = self.adain1(out, w)               # [B, C, H, W]  W : 바꿔야할지..?
        out = self.conv(out)                    # [B, C, H, W]
        out = self.noise_inject2(out, noise2)   # [B, C, H, W]
        out = self.adain2(out, w)               # [B, C, H, W]  W : 바꿔야할지..?
        return out


class SynthesisBlock(nn.Module):
    def __init__(self, w_dim, in_dim, out_dim):
        super(SynthesisBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = ConvBlock(in_dim, out_dim, ksize=3, stride=1, padding=1, n_slope=0.01, norm="SN",
                               non_linear='leaky_relu')
        self.noise_inject1 = NoiseInjection(in_dim=out_dim)
        self.adain1 = AdaptiveInstanceNorm(w_dim, out_dim)

        self.conv2 = ConvBlock(in_dim, out_dim, ksize=3, stride=1, padding=1, n_slope=0.01, norm="SN",
                               non_linear='leaky_relu')
        self.noise_inject2 = NoiseInjection(in_dim=out_dim)
        self.adain2 = AdaptiveInstanceNorm(w_dim, out_dim)

    def forward(self, x, w, noise):
        noise1, noise2 = torch.split(noise, 2, dim=1)   # [B, 1, H, W], [B, 1, H, W]
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.noise_inject1(out, noise1)
        out = self.adain1(out, w)

        out = self.conv2(out)
        out = self.noise_inject2(out, noise2)
        out = self.adain2(out, w)
        return out
    

class Generator(nn.Module):
    def __init__(self, out_dim_list, img_size_list, w_dim):
        super(Generator, self).__init__()
        self.img_size_list = img_size_list
        progress_blocks = []
        for idx, out_dim in enumerate(out_dim_list):
            if idx == 0:
                progress_blocks.append(SynthesisConstBlock(w_dim, 512, out_dim_list[0]))
            else:
                progress_blocks.append(SynthesisBlock(w_dim, out_dim_list[idx-1], out_dim_list[idx]))
        to_rgb_blocks = [ConvBlock(out_dim, 3, ksize=1, stride=1, padding=0) for out_dim in out_dim_list]

        self.progress = nn.ModuleList(*progress_blocks)
        self.to_rgb = nn.ModuleList(*to_rgb_blocks)

    def forward(self, x, w, step):
        for idx, progress_block, to_rgb in enumerate(zip(self.progress, self.to_rgb)):
            noise = torch.randn(x.size(0), 2, img_size_list[idx], img_size_list[idx])
            x = progress_block(x, w, noise)
            if idx == step:
                out = to_rgb(x)
                return out


class Discriminator(nn.Module):
    def __init__(self, out_dim_list):
        super(Discriminator, self).__init__()
        progress_blocks = []
        for idx, out_dim in enumerate(out_dim_list):
            if idx == 0:
                progress_blocks.append(DownscaleBlock(512, out_dim_list[idx]))
            else:
                progress_blocks.append(DownscaleBlock(out_dim_list[idx-1], out_dim_list[idx]))
        from_rgb_blocks = [ConvBlock(3, out_dim, ksize=1, stride=1, padding=0) for out_dim in out_dim_list]

        self.progress = nn.ModuleList(*progress_blocks)
        self.from_rgb = nn.ModuleList(*from_rgb_blocks)

        self.linear = nn.Linear(512, 1)

        self.n_layer = len(self.progress)

    def forward(self, x, step):
        for idx, progress_block, from_rgb in enumerate(zip(self.progress, self.from_rgb)):
            if idx == step:
                out = self.from_rgb(x)
        return out

