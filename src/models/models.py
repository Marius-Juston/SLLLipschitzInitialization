import torch.nn as nn
from torchvision.transforms import Normalize

from models.layers import LinearNormalized, PaddingChannels
from models.layers import SDPBasedLipschitzConvLayer, SDPBasedLipschitzLinearLayer


class NormalizedModel(nn.Module):

    def __init__(self, model, mean, std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.normalize = Normalize(mean, std)

    def forward(self, x):
        return self.model(self.normalize(x))


class LipschitzNetwork(nn.Module):

    def __init__(self, config, n_classes):
        super(LipschitzNetwork, self).__init__()

        self.n_conv = config.n_conv
        self.n_dense = config.n_dense
        self.cin = config.w
        self.conv_inner_dim = config.conv_inner_dim
        self.dense_inner_dim = config.dense_inner_dim
        self.n_classes = n_classes

        if config.dataset == 'tiny-imagenet':
            imsize = 64
        else:
            imsize = 32

        self.model = []
        self.model.append(
            PaddingChannels(self.cin, 3, "zero")
        )

        for _ in range(self.n_conv):
            self.model.append(SDPBasedLipschitzConvLayer(self.cin, self.conv_inner_dim))

        self.model.append(nn.AvgPool2d(4, divisor_override=4))

        self.model.append(nn.Flatten())
        if config.dataset in ['cifar10', 'cifar100']:
            in_channels = self.cin * 8 * 8
        elif config.dataset == 'tiny-imagenet':
            in_channels = self.cin * 16 * 16

        for _ in range(self.n_dense):
            self.model.append(SDPBasedLipschitzLinearLayer(in_channels, self.dense_inner_dim))

        self.model.append(LinearNormalized(in_channels, self.n_classes))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class LipschitzLinearNetwork(nn.Module):

    def __init__(self, in_features, out_features, n_dense=15, dense_inner_dim=256, bias_init=True, device=None, dtype=None):
        super(LipschitzLinearNetwork, self).__init__()

        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.n_dense = n_dense
        self.dense_inner_dim = dense_inner_dim
        self.out_features = out_features
        self.in_features = in_features

        self.model = []

        in_features = in_features

        for _ in range(self.n_dense):
            self.model.append(SDPBasedLipschitzLinearLayer(in_features, dense_inner_dim, bias_init=bias_init, **self.factory_kwargs))

            in_features = dense_inner_dim

        self.model.append(SDPBasedLipschitzLinearLayer(in_features, out_features, bias_init=bias_init, **self.factory_kwargs))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
