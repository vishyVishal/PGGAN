from .model_base import *


class GConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 nonlinearity,
                 use_pixelnorm=True):
        super(GConvBlock, self).__init__()
        layers = [EqualizedConv2d(in_channels, out_channels, kernel_size, padding=padding), nonlinearity]
        if use_pixelnorm:
            layers.append(PixelNormLayer())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 nonlinearity,
                 use_gdrop=True,
                 use_pixelnorm=False):
        super(DConvBlock, self).__init__()
        layers = []
        if use_gdrop:
            layers.append(GeneralizedDropout(mode='prop', strength=0.2))
        layers.extend([EqualizedConv2d(in_channels, out_channels, kernel_size, padding=padding), nonlinearity])
        if use_pixelnorm:
            layers.append(PixelNormLayer())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FromOrToRGBLayer(nn.Module):
    """
    A 1×1 convolution layer, which helps to convert between RGB and feature maps
    """
    def __init__(self, in_channels, out_channels, nonlinearity=None, use_pixelnorm=False):
        super(FromOrToRGBLayer, self).__init__()
        layers = [EqualizedConv2d(in_channels, out_channels, kernel_size=1)]
        if nonlinearity is not None:
            layers.append(nonlinearity)
        if use_pixelnorm:
            layers.append(PixelNormLayer())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self,
                 num_channels=3,
                 resolution=256,
                 latent_dim=512,
                 feature_map_base=4096,
                 feature_map_decay=1.0,
                 max_feature_map=256,
                 normalize_latent=True,  # whether or not to use pixel normalization to latent vector
                 use_pixelnorm=True,
                 use_leakyrelu=True,
                 negative_slope=0.2,
                 tanh_at_end=False  # whether or not to use tanh after each to_rgb layer
                 ):
        super(Generator, self).__init__()
        self.num_channels = num_channels
        self.resolution = resolution
        self.latent_dim = latent_dim
        self.feature_map_base = feature_map_base
        self.feature_map_decay = feature_map_decay
        self.max_feature_map = max_feature_map
        self.normalize_latent = normalize_latent
        self.use_pixelnorm = use_pixelnorm
        self.use_leakyrelu = use_leakyrelu
        self.tanh_at_end = tanh_at_end

        nonlinear = nn.LeakyReLU(negative_slope) if self.use_leakyrelu else nn.ReLU()
        out_active = nn.Tanh() if self.tanh_at_end else None
        self.to_rgb_layers = nn.ModuleList()
        self.progress_growing_layers = nn.ModuleList()

        first_layer = []
        if normalize_latent:
            first_layer.append(PixelNormLayer())
        first_layer.extend([
            ReshapeLayer((latent_dim, 1, 1)),
            GConvBlock(latent_dim, self.get_feature_map_number(1),
                       kernel_size=4, padding=3, nonlinearity=nonlinear, use_pixelnorm=use_pixelnorm),
            GConvBlock(self.get_feature_map_number(1), self.get_feature_map_number(1),
                       kernel_size=3, padding=1, nonlinearity=nonlinear, use_pixelnorm=use_pixelnorm)
        ])

        first_layer = nn.Sequential(*first_layer)
        self.progress_growing_layers.append(first_layer)
        self.to_rgb_layers.append(FromOrToRGBLayer(self.get_feature_map_number(1), num_channels,
                                                   nonlinearity=out_active, use_pixelnorm=False))
        self.R = int(math.log2(resolution))
        for r in range(2, self.R):
            in_channels, out_channels = self.get_feature_map_number(r - 1), self.get_feature_map_number(r)
            self.progress_growing_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                GConvBlock(in_channels, out_channels, kernel_size=3, padding=1,
                           nonlinearity=nonlinear, use_pixelnorm=use_pixelnorm),
                GConvBlock(out_channels, out_channels, kernel_size=3, padding=1,
                           nonlinearity=nonlinear, use_pixelnorm=use_pixelnorm)
            ))
            self.to_rgb_layers.append(FromOrToRGBLayer(out_channels, num_channels,
                                                       nonlinearity=out_active, use_pixelnorm=False))

    def get_feature_map_number(self, stage):
        return min(int(self.feature_map_base / (2.0 ** (stage * self.feature_map_decay))), self.max_feature_map)

    def forward(self, x, level, mode='stabilize', alpha=None):
        """
        level: 表示正在进行的分辨率的2底数对数,例如,当前为64 pixel时,level为6
        mode: 取值为'stabilize'或'transition',后者在当前level进行fade in
        """
        assert level in range(2, self.R + 1)
        assert mode in {'stabilize', 'transition'}
        if mode == 'stabilize':
            from_, to_ = 0, level - 1
            for i in range(from_, to_):
                x = self.progress_growing_layers[i](x)
            x = self.to_rgb_layers[to_ - 1](x)
            return x
        assert alpha is not None
        from_, to_ = 0, level - 2
        for i in range(from_, to_):
            x = self.progress_growing_layers[i](x)
        out1 = self.to_rgb_layers[to_ - 1](x)
        out1 = F.interpolate(out1, scale_factor=2, mode='nearest')
        x = self.progress_growing_layers[to_](x)
        out2 = self.to_rgb_layers[to_](x)
        out = (1 - alpha) * out1 + alpha * out2
        return out


class Discriminator(nn.Module):
    def __init__(self,
                 num_channels=3,
                 resolution=256,
                 feature_map_base=4096,
                 feature_map_decay=1.0,
                 max_feature_map=256,
                 negative_slope=0.2,
                 minibatch_stat_concat=True,
                 use_weightscale=True,
                 use_layernorm=False,
                 use_gdrop=True,
                 sigmoid_at_end=False):
        super(Discriminator, self).__init__()
        self.num_channels = num_channels
        self.resolution = resolution
        self.feature_map_base = feature_map_base
        self.feature_map_decay = feature_map_decay
        self.max_feature_map = max_feature_map
        self.minibatch_stat_concat = minibatch_stat_concat
        self.use_weightscale = use_weightscale
        self.use_layernorm = use_layernorm
        self.use_gdrop = use_gdrop
        self.sigmoid_at_end = sigmoid_at_end

        nonlinear = nn.LeakyReLU(negative_slope)
        out_active = nn.Sigmoid() if self.sigmoid_at_end else None
        self.R = int(math.log2(resolution))

        self.from_rgb_layers = nn.ModuleList()
        self.progress_growing_layers = nn.ModuleList()
        for r in range(self.R - 1, 1, -1):
            in_channels, out_channels = self.get_feature_map_number(r), self.get_feature_map_number(r - 1)
            self.from_rgb_layers.append(FromOrToRGBLayer(num_channels, in_channels, nonlinearity=nonlinear, use_pixelnorm=False))
            self.progress_growing_layers.append(nn.Sequential(
                DConvBlock(in_channels, in_channels, kernel_size=3, padding=1,
                           nonlinearity=nonlinear, use_gdrop=use_gdrop),
                DConvBlock(in_channels, out_channels, kernel_size=3, padding=1,
                           nonlinearity=nonlinear, use_gdrop=use_gdrop),
                nn.AvgPool2d(kernel_size=2, stride=2, count_include_pad=False)
            ))
        last_layers = []
        in_channels, out_channels = self.get_feature_map_number(1), self.get_feature_map_number(1)
        self.from_rgb_layers.append(FromOrToRGBLayer(num_channels, in_channels, nonlinearity=nonlinear, use_pixelnorm=False))
        if minibatch_stat_concat:
            last_layers.append(MinibatchStatConcatLayer())
            in_channels += 1
        last_layers.extend([
            DConvBlock(in_channels, out_channels, kernel_size=3, padding=1,
                       nonlinearity=nonlinear, use_gdrop=use_gdrop),
            DConvBlock(out_channels, self.get_feature_map_number(0), kernel_size=4, padding=0,
                       nonlinearity=nonlinear, use_gdrop=use_gdrop),
            Flatten(),
            nn.Linear(self.get_feature_map_number(0), 128),
            nn.LeakyReLU(negative_slope),
            nn.Linear(128, 1)
        ])
        if sigmoid_at_end:
            last_layers.append(out_active)
        self.progress_growing_layers.append(nn.Sequential(*last_layers))

    def get_feature_map_number(self, stage):
        return min(int(self.feature_map_base / (2.0 ** (stage * self.feature_map_decay))), self.max_feature_map)

    def forward(self, x, level, mode='stabilize', alpha=None):
        """
        level: 表示正在进行的分辨率的2底数对数,例如,当前为64 pixel时,level为6
        mode: 取值为'stabilize'或'transition',后者在当前level进行fade in
        """
        assert level in range(2, self.R + 1)
        assert mode in {'stabilize', 'transition'}
        if mode == 'stabilize':
            from_, to_ = self.R - level, self.R - 1
            x = self.from_rgb_layers[from_](x)
            for i in range(from_, to_):
                x = self.progress_growing_layers[i](x)
            return x
        assert alpha is not None
        from_, to_ = self.R - level + 1, self.R - 1
        in1 = F.avg_pool2d(x, kernel_size=2, stride=2)
        in1 = self.from_rgb_layers[from_](in1)
        in2 = self.from_rgb_layers[from_ - 1](x)
        in2 = self.progress_growing_layers[from_ - 1](in2)
        out = (1 - alpha) * in1 + alpha * in2
        for i in range(from_, to_):
            out = self.progress_growing_layers[i](out)
        return out
