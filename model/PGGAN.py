from .model_base import *


class GConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 nonlinearity,
                 nonlinearity_name,
                 param=None,
                 use_weightscale=True,
                 use_batchnorm=False,
                 use_pixelnorm=True):
        super(GConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)]
        kaiming_init(layers[-1], nonlinearity=nonlinearity_name, param=param)
        if use_weightscale:
            layers.append(WeightScaleLayer(layers[-1]))
        layers.append(nonlinearity)
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
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
                 nonlinearity_name,
                 param=None,
                 use_weightscale=True):
        super(DConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)]
        kaiming_init(layers[-1], nonlinearity=nonlinearity_name, param=param)
        if use_weightscale:
            layers.append(WeightScaleLayer(layers[-1]))
        layers.append(nonlinearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FromOrToRGBLayer(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinearity=None,
                 nonlinearity_name=None, param=None, use_weightscale=True):
        super(FromOrToRGBLayer, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)]
        kaiming_init(layers[-1], nonlinearity=nonlinearity_name, param=param)
        if use_weightscale:
            layers.append(WeightScaleLayer(layers[-1]))
        if nonlinearity is not None:
            layers.append(nonlinearity)
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
                 use_weightscale=True,
                 normalize_latent=True,
                 use_pixelnorm=True,
                 use_leakyrelu=True,
                 negative_slope=0.2,
                 use_batchnorm=True,
                 tanh_at_end=True):
        super(Generator, self).__init__()
        self.num_channels = num_channels
        self.resolution = resolution
        self.latent_dim = latent_dim
        self.feature_map_base = feature_map_base
        self.feature_map_decay = feature_map_decay
        self.max_feature_map = max_feature_map
        self.use_weightscale = use_weightscale
        self.normalize_latent = normalize_latent
        self.use_pixelnorm = use_pixelnorm
        self.use_leakyrelu = use_leakyrelu
        self.use_batchnorm = use_batchnorm
        self.tanh_at_end = tanh_at_end
        self.up_sampler = nn.Upsample(scale_factor=2, mode='nearest')

        nonlinear = nn.LeakyReLU(negative_slope) if self.use_leakyrelu else nn.ReLU()
        nonlinearity_name = 'leaky_relu' if self.use_leakyrelu else 'relu'
        out_active = nn.Tanh() if self.tanh_at_end else None
        out_active_name = 'tanh' if self.tanh_at_end else 'linear'
        self.to_rgb_layers = nn.ModuleList()
        self.progress_growing_layers = nn.ModuleList()

        first_layer = nn.Sequential(
            ReshapeLayer((latent_dim, 1, 1)),
            GConvBlock(latent_dim, self.get_feature_map_number(1),
                       kernel_size=4, padding=3, nonlinearity=nonlinear,
                       nonlinearity_name=nonlinearity_name, param=negative_slope,
                       use_weightscale=use_weightscale,
                       use_batchnorm=use_batchnorm, use_pixelnorm=use_pixelnorm),
            GConvBlock(self.get_feature_map_number(1), self.get_feature_map_number(1),
                       kernel_size=3, padding=1, nonlinearity=nonlinear,
                       nonlinearity_name=nonlinearity_name, param=negative_slope,
                       use_weightscale=use_weightscale,
                       use_batchnorm=use_batchnorm, use_pixelnorm=use_pixelnorm)
        )
        self.progress_growing_layers.append(first_layer)
        self.to_rgb_layers.append(FromOrToRGBLayer(self.get_feature_map_number(1), num_channels,
                                                   nonlinearity=out_active, nonlinearity_name=out_active_name,
                                                   use_weightscale=use_weightscale))
        self.R = int(math.log2(resolution))
        for r in range(2, self.R):
            in_channels, out_channels = self.get_feature_map_number(r - 1), self.get_feature_map_number(r)
            self.progress_growing_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                GConvBlock(in_channels, out_channels, kernel_size=3, padding=1,
                           nonlinearity=nonlinear, nonlinearity_name=nonlinearity_name, param=negative_slope,
                           use_weightscale=use_weightscale,
                           use_batchnorm=use_batchnorm, use_pixelnorm=use_pixelnorm),
                GConvBlock(out_channels, out_channels, kernel_size=3, padding=1,
                           nonlinearity=nonlinear, nonlinearity_name=nonlinearity_name, param=negative_slope,
                           use_weightscale=use_weightscale,
                           use_batchnorm=use_batchnorm, use_pixelnorm=use_pixelnorm)
            ))
            self.to_rgb_layers.append(FromOrToRGBLayer(out_channels, num_channels, nonlinearity=out_active,
                                                       nonlinearity_name=out_active_name,
                                                       use_weightscale=use_weightscale))

    def get_feature_map_number(self, stage):
        return min(int(self.feature_map_base / (2.0 ** (stage * self.feature_map_decay))), self.max_feature_map)

    def forward(self, x, level=None, mode='stabilize', alpha=None):
        """
        level: 表示正在进行的分辨率的2底数对数,例如,当前为64 pixel时,level为6
        mode: 取值为'stabilize'或'transition',后者在当前level进行fade in
        """
        if level is None:
            level = self.R
        assert level in range(2, self.R + 1)
        assert mode in {'stabilize', 'transition'}
        from_, to_ = 0, level - 1
        if mode == 'stabilize':
            for i in range(from_, to_):
                x = self.progress_growing_layers[i](x)
            x = self.to_rgb_layers[to_ - 1](x)
            return x
        assert alpha is not None
        from_, to_ = 0, level - 2
        for i in range(from_, to_):
            x = self.progress_growing_layers[i](x)
        out1 = self.up_sampler(x)
        out1 = self.to_rgb_layers[to_ - 1](out1)
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
        self.sigmoid_at_end = sigmoid_at_end
        self.down_sampler = nn.AvgPool2d(kernel_size=2)

        nonlinear = nn.LeakyReLU(negative_slope)
        out_active = nn.Sigmoid() if self.sigmoid_at_end else None
        self.R = int(math.log2(resolution))

        self.from_rgb_layers = nn.ModuleList()
        self.progress_growing_layers = nn.ModuleList()
        for r in range(self.R - 1, 1, -1):
            in_channels, out_channels = self.get_feature_map_number(r), self.get_feature_map_number(r - 1)
            self.from_rgb_layers.append(FromOrToRGBLayer(num_channels, in_channels, nonlinearity=nonlinear,
                                                         nonlinearity_name='leaky_relu', param=negative_slope,
                                                         use_weightscale=use_weightscale))
            self.progress_growing_layers.append(nn.Sequential(
                DConvBlock(in_channels, in_channels, kernel_size=3, padding=1,
                           nonlinearity=nonlinear, nonlinearity_name='leaky_relu', param=negative_slope,
                           use_weightscale=use_weightscale),
                DConvBlock(in_channels, out_channels, kernel_size=3, padding=1,
                           nonlinearity=nonlinear, nonlinearity_name='leaky_relu', param=negative_slope,
                           use_weightscale=use_weightscale),
                nn.AvgPool2d(kernel_size=2, stride=2, count_include_pad=False)
            ))
        last_layers = []
        in_channels, out_channels = self.get_feature_map_number(1), self.get_feature_map_number(1)
        self.from_rgb_layers.append(FromOrToRGBLayer(num_channels, in_channels, nonlinearity=nonlinear,
                                                     nonlinearity_name='leaky_relu', param=negative_slope,
                                                     use_weightscale=use_weightscale))
        if minibatch_stat_concat:
            last_layers.append(MinibatchStatConcatLayer())
            in_channels += 1
        last_layers.append(nn.Sequential(
            DConvBlock(in_channels, out_channels, kernel_size=3, padding=1,
                       nonlinearity=nonlinear, nonlinearity_name='leaky_relu', param=negative_slope,
                       use_weightscale=use_weightscale),
            DConvBlock(out_channels, self.get_feature_map_number(0), kernel_size=4, padding=0,
                       nonlinearity=nonlinear, nonlinearity_name='leaky_relu', param=negative_slope,
                       use_weightscale=use_weightscale),
            Flatten(),
            nn.Linear(self.get_feature_map_number(0), 1)
        ))
        if sigmoid_at_end:
            last_layers.append(out_active)
        self.progress_growing_layers.append(nn.Sequential(*last_layers))

    def get_feature_map_number(self, stage):
        return min(int(self.feature_map_base / (2.0 ** (stage * self.feature_map_decay))), self.max_feature_map)

    def forward(self, x, level=None, mode='stabilize', alpha=None):
        """
        level: 表示正在进行的分辨率的2底数对数,例如,当前为64 pixel时,level为6
        mode: 取值为'stabilize'或'transition',后者在当前level进行fade in
        """
        if level is None:
            level = self.R
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
        in1 = self.down_sampler(x)
        in1 = self.from_rgb_layers[from_](in1)
        in2 = self.from_rgb_layers[from_ - 1](x)
        in2 = self.progress_growing_layers[from_ - 1](in2)
        out = (1 - alpha) * in1 + alpha * in2
        for i in range(from_, to_):
            out = self.progress_growing_layers[i](out)
        return out
