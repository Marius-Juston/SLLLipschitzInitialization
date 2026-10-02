"""Paper-oriented feedforward layers and pinned upstream residual control."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .math_checks import mean_field
from .vendor import sll_layers as upstream

UPSTREAM_COMMIT = '87865dbf74f07b443b595ea777edbb1d4596faf0'


def inverse_positive(x, power):
    # Do not evaluate reciprocal at zero: avoids NaN backward through masking.
    return torch.where(x > 0, x.clamp_min(torch.finfo(x.dtype).tiny).pow(power), torch.zeros_like(x))


class ColumnLinear(nn.Module):
    def __init__(self, inputs, outputs, bias_mode='zero', train_q=False, activation=True, bias_seed=0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(outputs, inputs))
        self.log_q = nn.Parameter(torch.zeros(inputs), requires_grad=train_q)
        self.bias = nn.Parameter(torch.empty(outputs))
        self.activation = activation
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        with torch.no_grad():
            # Separate RNG stream: varying bias mode does not change later weights.
            gen = torch.Generator().manual_seed(torch.initial_seed() + 100003 + bias_seed)
            if bias_mode == 'zero':
                self.bias.zero_()
            elif bias_mode == 'default':
                self.bias.uniform_(-1/math.sqrt(inputs), 1/math.sqrt(inputs), generator=gen)
            elif bias_mode == 'corrected':
                r = .5 * inputs * mean_field(outputs, inputs)
                self.bias.normal_(std=math.sqrt(max(0., 1-r)), generator=gen)
            else:
                raise ValueError(bias_mode)

    def normalized_weight(self):
        # W is outputs x inputs; normalization is W T^{-1/2}, as in paper.
        gram = (self.weight.T @ self.weight).abs()
        delta = self.log_q[None,:] - self.log_q[:,None]
        t = (gram * delta.exp()).sum(dim=1)
        return self.weight * inverse_positive(t, -.5)[None,:]

    def forward(self, x):
        y = F.linear(x, self.normalized_weight(), self.bias)
        return F.relu(y) if self.activation else y


class FeedForward(nn.Module):
    def __init__(self, inputs=3072, classes=10, depth=5, width=256, bias_mode='zero', train_q=False):
        super().__init__()
        layers = [nn.Flatten()]
        for i in range(depth):
            layers.append(ColumnLinear(inputs if i == 0 else width, width, bias_mode, train_q, bias_seed=i))
        layers.append(ColumnLinear(width, classes, bias_mode, train_q, activation=False, bias_seed=depth))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def certificate(self, logits, labels):
        other = logits.clone().scatter_(1, labels[:,None], -torch.inf).amax(1)
        return (logits.gather(1, labels[:,None]).squeeze(1)-other).clamp_min(0)/math.sqrt(2)


class ResidualSmall(nn.Module):
    """Official small architecture; upstream head uses pairwise certificates.

    Training input mean shift is 1-Lipschitz. Backbone is 1-Lipschitz;
    head rows have norm 1, but its vector norm need not equal 1.
    """
    def __init__(self, classes=10, bias_mode='default'):
        super().__init__()
        self.register_buffer('mean', torch.tensor([.4914,.4822,.4465])[None,:,None,None])
        layers = [upstream.PaddingChannels(45,3,'zero')]
        layers += [upstream.SDPBasedLipschitzConvLayer(45,5) for _ in range(20)]
        layers += [nn.AvgPool2d(4, divisor_override=4), nn.Flatten()]
        layers += [upstream.SDPBasedLipschitzLinearLayer(45*64,2048) for _ in range(7)]
        self.backbone = nn.Sequential(*layers)
        self.head = upstream.LinearNormalized(45*64,classes)
        if bias_mode == 'corrected':
            # Empirical transfer of feedforward variance correction, not a residual theorem.
            for index, layer in enumerate(self.backbone):
                if isinstance(layer, (upstream.SDPBasedLipschitzConvLayer, upstream.SDPBasedLipschitzLinearLayer)):
                    w = layer.kernel if hasattr(layer,'kernel') else layer.weight
                    inputs, outputs = w[0].numel(), w.shape[0]
                    gen = torch.Generator().manual_seed(torch.initial_seed()+200003+index)
                    with torch.no_grad():
                        layer.bias.normal_(std=math.sqrt(max(0.,1-.5*inputs*mean_field(outputs,inputs))), generator=gen)
        elif bias_mode != 'default':
            raise ValueError('Residual control supports default/corrected bias')

    def forward(self, x):
        return self.head(self.backbone(x-self.mean))

    def certificate(self, logits, labels):
        q = F.normalize(self.head.weight,dim=1)
        distances = (q[labels,None,:]-q[None,:,:]).norm(dim=2)
        margins = logits.gather(1,labels[:,None])-logits
        # Identical head rows give a constant margin: +infinity if positive, zero otherwise.
        radii = torch.where(distances > 0, margins/distances.clamp_min(1e-30),
                            torch.where(margins > 0, torch.inf, 0.))
        radii.scatter_(1,labels[:,None],torch.inf)
        return radii.amin(dim=1).clamp_min(0)


def build(config, inputs, classes):
    if config['model'] == 'residual':
        return ResidualSmall(classes,config['bias'])
    return FeedForward(inputs,classes,config['depth'],config['width'],config['bias'],config['model']=='sll')
