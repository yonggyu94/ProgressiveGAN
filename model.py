import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import math


class WrongNormException(Exception):
    def __str__(self):
        return 'You should choose \'BN\', \'IN\' or \'SN\''


class WrongNonLinearException(Exception):
    def __str__(self):
        return 'You should choose \'relu\', \'leaky_relu\' or \'tanh\''


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class EqualizedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride, padding):
        super(EqualizedConv2d, self).__init__()
        conv = nn.Conv2d(in_ch, out_ch, k_size, stride, padding)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, x):
        out = self.conv(x)
        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x):
        out = x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)
        return out


class StdConcat(nn.Module):
    def __init__(self):
        super(StdConcat, self).__init__()

    def forward(self, x):
        mean_std = torch.mean(x.std(0))
        mean_std = mean_std.expand(x.size(0), 1, 4, 4)
        out = torch.cat([x, mean_std], dim=1)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ksize=3, stride=1, padding=1, n_slope=0.01,
                 norm='SN', non_linear='leaky_relu', equalized=True):
        super(ConvBlock, self).__init__()
        layers = []

        # Select normalization
        if norm == 'SN':
            layers.append(spectral_norm(nn.Conv2d(in_dim, out_dim, kernel_size=ksize, stride=stride, padding=padding)))
        else:
            if equalized:
                layers.append(EqualizedConv2d(in_dim, out_dim, ksize, stride, padding))
            else:
                layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=ksize, stride=stride, padding=padding))

            if norm == 'BN':
                layers.append(nn.BatchNorm2d(out_dim, affine=True))
            elif norm == 'IN':
                layers.append(nn.InstanceNorm2d(out_dim, affien=True))
            elif norm == 'PN':
                layers.append(PixelNorm())
            elif norm == None : pass
            else: raise WrongNormException

        # Select activation function
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


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, ksize_1, pad_1, ksize_2, pad_2, norm, n_slope=0.2, non_linear='leaky_relu',
                 equlized=True):
        super(Block, self).__init__()
        self.convblock_1 = ConvBlock(in_dim, out_dim, ksize_1, padding=pad_1,
                                     n_slope=n_slope, norm=norm, non_linear=non_linear, equalized=equlized)
        self.convblock_2 = ConvBlock(out_dim, out_dim, ksize_2, padding=pad_2,
                                     n_slope=n_slope, norm=norm, non_linear=non_linear, equalized=equlized)

    def forward(self, x):
        out = self.convblock_1(x)
        out = self.convblock_2(out)
        return out


class Generator(nn.Module):
    def __init__(self, channel_list=[512, 512, 512, 512, 256, 128, 64, 32], n_label=10, n_slope=0.2):
        super(Generator, self).__init__()

        # self.label_embed = nn.Embedding(n_label, n_label)
        # self.label_embed.weight.data.normal_()
        self.code_norm = PixelNorm()

        # self.progress = nn.ModuleList([Block(ch_list[0], ch_list[0], 4, 3, 3, 1, norm="PN"),  # [B, 512, 4, 4]
        #                                Block(ch_list[0], ch_list[1], 3, 1, 3, 1, norm="PN"),  # [B, 512, 8, 8]
        #                                Block(ch_list[1], ch_list[2], 3, 1, 3, 1, norm="PN"),  # [B, 512, 16, 16]
        #                                Block(ch_list[2], ch_list[3], 3, 1, 3, 1, norm="PN"),  # [B, 512, 32, 32]
        #                                Block(ch_list[3], ch_list[4], 3, 1, 3, 1, norm="PN"),  # [B, 256, 64, 64]
        #                                Block(ch_list[4], ch_list[5], 3, 1, 3, 1, norm="PN"),  # [B, 128, 128, 128]
        #                                Block(ch_list[5], ch_list[6], 3, 1, 3, 1, norm="PN"),  # [B, 64, 256, 256]
        #                                Block(ch_list[6], ch_list[7], 3, 1, 3, 1, norm="PN")   # [B, 32, 512, 512]
        #                                ])

        # self.to_rgb = nn.ModuleList([nn.Conv2d(512, 3, 1),
        #                              nn.Conv2d(512, 3, 1),
        #                              nn.Conv2d(512, 3, 1),
        #                              nn.Conv2d(512, 3, 1),
        #                              nn.Conv2d(256, 3, 1),
        #                              nn.Conv2d(128, 3, 1),
        #                              nn.Conv2d(64, 3, 1),
        #                              nn.Conv2d(32, 3, 1)
        #                              ])

        progress_layers = []
        to_rgb_layers = []
        for i in range(len(channel_list)):
            if i == 0:
                progress_layers.append(Block(channel_list[i], channel_list[i], 4, 3, 3, 1, norm="PN"))
            else:
                progress_layers.append(Block(channel_list[i - 1], channel_list[i], 3, 1, 3, 1, norm="PN"))
            to_rgb_layers.append(nn.Conv2d(channel_list[i], 3, 1))

        self.progress = nn.ModuleList(progress_layers)
        self.to_rgb = nn.ModuleList(to_rgb_layers)

    def forward(self, x, step=0, alpha=-1):
        out = self.code_norm(x)
        # label = self.label_embed(label)
        # out = torch.cat([input, label], 1).unsqueeze(2).unsqueeze(3)

        for i, (block, to_rgb) in enumerate(zip(self.progress, self.to_rgb)):
            if i > 0:
                upsample = F.upsample(out, scale_factor=2)
                out = block(upsample)
            else:
                out = block(out)

            if i == step:
                out = to_rgb(out)
                if i != 0 and 0 <= alpha < 1:               # The first module does not need previous to_rgb module
                    skip_rgb = self.to_rgb[i-1](upsample)
                    out = (1-alpha)*skip_rgb + alpha*out
                break
        return out


class Discriminator(nn.Module):
    def __init__(self, channel_list=[512, 512, 512, 512, 256, 128, 64, 32], n_slope=0.2, n_label=10):
        super(Discriminator, self).__init__()
        reversed(channel_list)

        self.std_concat = StdConcat()

        progress_layers = []
        from_rgb_layers = []
        for i in range(len(channel_list) - 1, -1, -1):
            if i == 0:
                progress_layers.append(Block(channel_list[i] + 1, channel_list[i], 3, 1, 4, 0, norm="SN", equlized=False))
            else:
                progress_layers.append(Block(channel_list[i], channel_list[i - 1], 3, 1, 3, 1, norm="SN", equlized=False))
            from_rgb_layers.append(nn.Conv2d(3, channel_list[i], 1))

        self.progress = nn.ModuleList(progress_layers)
        self.from_rgb = nn.ModuleList(from_rgb_layers)

        self.n_layer = len(self.progress)
        self.linear = nn.Linear(512, 1)

    def forward(self, x, step=0, alpha=-1):
        step = self.n_layer - 1 - step
        for i in range(step, self.n_layer):
            if i == step:
                out = self.from_rgb[i](x)

            if i == (self.n_layer-1):
                out = self.std_concat(out)
                out = self.progress[i](out)
            else:
                out = self.progress[i](out)
                out = F.avg_pool2d(out, 2)

            if i == step:
                if i != 7 and 0 <= alpha < 1:
                    downsample = F.avg_pool2d(x, 2)
                    skip_rgb = self.from_rgb[i+1](downsample)
                    out = (1-alpha)*skip_rgb + alpha*out

        out = out.squeeze(3).squeeze(2)
        out = self.linear(out)

        return out[:, 0]


if __name__ == "__main__":
    z = torch.rand(4, 512, 1, 1)
    img = torch.rand(4, 3, 512, 512)

    g = Generator(channel_list=[512, 512, 512, 512, 256, 128, 64, 32])
    d = Discriminator(channel_list=[512, 512, 512, 512, 256, 128, 64, 32])
    alpha = 0.5

    print("Generator")
    for step in range(8):
        out = g(z, step=step, alpha=alpha)
        print(out.shape)
    print("Discriminator")
    out = d(img, step=7, alpha=alpha)
    print(out.shape)
