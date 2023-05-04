import torch
import torch.nn as nn


class EasingModule():
    def __init__(self, easing_steps=5):
        super().__init__()

        self.easing_step = 0
        self.easing_steps = easing_steps

    def step(self):
        self.easing_step = min(self.easing_step + 1, self.easing_steps)

    def reset(self):
        self.easing_step = 0
    
    def eval(self):
        self.easing_step = self.easing_steps
    
    def combine(self, easing_out, easing_in):
        alfa = (self.easing_step + 1) / (self.easing_steps + 1)

        return easing_in * alfa + easing_out * (1 - alfa)


class PixelNorm2d(nn.Module):
    """Normalize all pixels along feature vector
    b[x,y] = a[x,y] / sqrt(1/N * sum(a[j, x,y]^2) + eps)
    j = 0..N, N = number of features
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * torch.rsqrt(torch.mean(torch.square(x), dim=1, keepdim=True) + 1e-8)
        return x


class StdDevConcat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        std = torch.std(x, dim=[0], keepdim=True)
        std = torch.mean(std, dim=[1, 2, 3], keepdim=True)
        std = std.repeat(batch_size, 1, height, width)

        return torch.cat([x, std], dim=1)


# class WScaleConv2d(nn.Module):
    # def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, gain=torch.sqrt(torch.tensor([2]))):
    #     super().__init__()

    #     self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    #     fan_in = torch.prod(torch.tensor(self.conv.weight.shape[:-1]))
    #     std = gain / torch.sqrt(fan_in)

    #     self.scale = torch.nn.Parameter(std)

    #     self.conv.weight.data.normal_(0, 1)
    #     self.conv.bias.data.zero_()

    # def forward(self, x):
    #     weight = self.conv.weight * self.scale
    #     return self.conv(x, weight, self.conv.bias)

# def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
#     if fan_in is None: fan_in = np.prod(shape[:-1])
#     std = gain / np.sqrt(fan_in) # He init
#     if use_wscale:
#         wscale = tf.constant(np.float32(std), name='wscale')
#         return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
#     else:
#         return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

# def get_weight(shape, gain=torch.sqrt(torch.tensor([2])), use_wscale=False, fan_in=None):
#     if fan_in is None: fan_in = torch.prod(torch.tensor(shape[:-1]))
#     std = gain / torch.sqrt(fan_in) # He init
#     if use_wscale:
#         wscale = nn.Parameter(std)
#         # return nn.Parameter(torch.randn(shape) * wscale)
#         return nn.Parameter(torch.randn(shape)) * wscale
#     else:
#         return nn.Parameter(torch.randn(shape) * std)

# def dense(x, fmaps, gain=np.sqrt(2), use_wscale=False):
#     if len(x.shape) > 2:
#         x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
#     w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
#     w = tf.cast(w, x.dtype)
#     return tf.matmul(x, w)

# def linear(x, fmaps, gain=torch.sqrt(torch.tensor([2])), use_wscale=False):
#     if len(x.shape) > 2:
#         x = x.reshape(x.shape[0], -1)
#     w = get_weight([x.shape[1], fmaps], gain=gain, use_wscale=use_wscale)
#     return torch.matmul(x, w)

# class Conv2D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, gain=torch.sqrt(torch.tensor([2])), use_wscale=False):
#         super().__init__()

#         self.weight = get_weight([out_channels, in_channels, kernel_size, kernel_size], gain=gain, use_wscale=use_wscale)
#         self.bias = nn.Parameter(torch.zeros(out_channels))

#         self.stride = stride
#         self.padding = padding

#     def forward(self, x):
#         return F.conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)


class Generator(EasingModule, nn.Module):
    # lat	4x4
    # lat	8x8
    # 256	16x16
    # 128	32x32
    # 64	64x64
    # 32	128x128
    # 16	256x256

    def __init__(self, latent_dim, easing_steps=5):
        super().__init__(easing_steps=easing_steps)

        block0 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            PixelNorm2d(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm2d()
        )
        to_rgb0 = nn.Conv2d(latent_dim, 3, kernel_size=1, stride=1, padding=0)

        block1 = nn.Sequential(
            nn.Upsample(scale_factor=2,  mode='nearest'),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm2d(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm2d()
        )
        to_rgb1 = nn.Conv2d(latent_dim, 3, kernel_size=1, stride=1, padding=0)

        block2 = nn.Sequential(
            nn.Upsample(scale_factor=2,  mode='nearest'),
            nn.Conv2d(latent_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm2d(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm2d()            
        )
        to_rgb2 = nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0)

        block3 = nn.Sequential(
            nn.Upsample(scale_factor=2,  mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm2d(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm2d()
        )
        to_rgb3 = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0)

        block4 = nn.Sequential(
            nn.Upsample(scale_factor=2,  mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm2d(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm2d()
        )
        to_rgb4 = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)

        block5 = nn.Sequential(
            nn.Upsample(scale_factor=2,  mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm2d(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm2d()
        )
        to_rgb5 = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0)

        block6 = nn.Sequential(
            nn.Upsample(scale_factor=2,  mode='nearest'),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm2d(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm2d()
        )
        to_rgb6 = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)

        self.blocks = nn.ModuleList([block0, block1, block2, block3, block4, block5, block6])
        self.to_rgbs = nn.ModuleList([to_rgb0, to_rgb1, to_rgb2, to_rgb3, to_rgb4, to_rgb5, to_rgb6])
        self.upsample = nn.Upsample(scale_factor=2,  mode='nearest')

        self.active_level = 0

    def increase_size(self):
        self.active_level += 1
        self.reset()

    def eval(self, size=6):
        while self.active_level != size:
            self.increase_size()

        super().eval()

    def forward(self, x):
        for i in range(self.active_level + 1):
            x = self.blocks[i](x)

            if self.easing_step != self.easing_steps and i == self.active_level - 1:
                to_combine = self.upsample(x.clone())
                to_combine = self.to_rgbs[i](to_combine)

            if i == self.active_level:
                x = self.to_rgbs[i](x)

        if self.active_level > 0 and self.easing_step != self.easing_steps:
            x = self.combine(to_combine, x)

        return x


class Discriminator(EasingModule, nn.Module):
    # 16	256x256
    # 32	128x128
    # 64	64x64
    # 128	32x32
    # 256	16x16
    # lat	8x8
    # lat	4x4

    def __init__(self, latent_dim, easing_steps=5):
        super().__init__(easing_steps=easing_steps)

        from_rgb6 = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        block6 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2, 2)
        )

        from_rgb5 = nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0)
        block5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2, 2)
        )

        from_rgb4 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
        block4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2, 2)
        )

        from_rgb3 = nn.Conv2d(3, 128, kernel_size=1, stride=1, padding=0)
        block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2, 2)
        )

        from_rgb2 = nn.Conv2d(3, 256, kernel_size=1, stride=1, padding=0)
        block2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2, 2)
        )

        from_rgb1 = nn.Conv2d(3, latent_dim, kernel_size=1, stride=1, padding=0)
        block1 = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2, 2)
        )

        from_rgb0 = nn.Conv2d(3, latent_dim, kernel_size=1, stride=1, padding=0)
        block0 = nn.Sequential(
            # StdDevConcat(),
            # nn.Conv2d(latent_dim + 1, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
        )

        self.blocks = nn.ModuleList([block0, block1, block2, block3, block4, block5, block6])
        self.from_rgbs = nn.ModuleList([from_rgb0, from_rgb1, from_rgb2, from_rgb3, from_rgb4, from_rgb5, from_rgb6])
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dim, 1)
        )
        self.downsample = nn.AvgPool2d(2, 2)

        self.active_level = 0

    def increase_size(self):
        self.active_level += 1
        self.reset()
    
    def eval(self, size):
        while self.active_level != size:
            self.increase_size()

        super().eval()

    def forward(self, x):
        if self.active_level > 0 and self.easing_step != self.easing_steps:
            to_combine = self.downsample(x.clone())
            to_combine = self.from_rgbs[self.active_level - 1](to_combine)

        x = self.from_rgbs[self.active_level](x)
        x = self.blocks[self.active_level](x)

        if self.active_level > 0 and self.easing_step != self.easing_steps:
            x = self.combine(to_combine, x)

        for i in range(1, self.active_level + 1):
            x = self.blocks[self.active_level - i](x)

        x = self.linear(x)

        return x
