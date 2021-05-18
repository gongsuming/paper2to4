# Non-local block using embedded gaussian
# Code from
# https://github.com/AlexHex7/Non-local_pytorch/blob/master/Non-Local_pytorch_0.3.1/lib/non_local_embedded_gaussian.py
import torch
from torch import nn
from torch.nn import functional as F

'''class Temporal(nn.Module):
    def __init__(self):
        super(Temporal, self).__init__()

    def forward(self, x):

        bt, c, h, w = x.size()                           # t = segment = 8
        q = x.view(bt // 8, 8, -1)                       #[b, t, chw]
        k = x.view(bt // 8, 8, -1).permute(0, 2, 1)      #[b, chw, t]
        v = x.view(bt // 8, 8, -1)                       #[b, t, chw]

        f = torch.matmul(q, k)
        f = F.softmax(f, dim=-1)                         #[b, t, t]

        out = torch.matmul(f, v)                         #[b, t, chw]
        out = out.view(-1, c, h, w)                      #[bt, c, h, w]     temporal-attention end
        out = out + x
        return  out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class PSPModule(nn.Module):
    def __init__(self, sizes=(1, 3, 6, 8)):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])

    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center

class ChannelAttention(nn.Module):
    def __init__(self, k_size=5):
        super(ChannelAttention, self).__init__()
        self.pool = PSPModule()
        self.conv = nn.Conv1d(110, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y1 = self.pool(x)
        y2 = y1.transpose(-1, -2)
        y3 = self.conv(y2)
        y4 = y3.transpose(-1, -2)
        y5 = y4.unsqueeze(-1)
        y = self.sigmoid(y5)
        return x * y.expand_as(x)   '''

class SpatialTemporal(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialTemporal, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)               # [bt, c, h, w]     spatial-attention end

        bt, c, h, w = x.size()           # t = segment = 8
        q = x.view(bt // 8, 8, -1)                       #[b, t, chw]
        k = x.view(bt // 8, 8, -1).permute(0, 2, 1)      #[b, chw, t]
        v = x.view(bt // 8, 8, -1)                       #[b, t, chw]

        f = torch.matmul(q, k)
        f = F.softmax(f, dim=-1)                         #[b, t, t]

        out = torch.matmul(f, v)                         #[b, t, chw]
        out = out.view(-1, c, h, w)                      #[bt, c, h, w]     temporal-attention end
        out = out + x

        return  out               


class PSPModule(nn.Module):
    def __init__(self, sizes=(1, 3, 6, 8)):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])

    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center

class ChannelTemporal(nn.Module):
    def __init__(self, k_size=5):
        super(ChannelTemporal, self).__init__()
        self.pool = PSPModule()
        self.conv = nn.Conv1d(110, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.pool(x)
        y2 = y1.transpose(-1, -2)
        y3 = self.conv(y2)
        y4 = y3.transpose(-1, -2)
        y5 = y4.unsqueeze(-1)
        y = self.sigmoid(y5)

        bt, c, h, w = y.size()                           # t = segment = 8
        q = y.view(bt // 8, 8, -1)                       #[b, t, chw]
        k = y.view(bt // 8, 8, -1).permute(0, 2, 1)      #[b, chw, t]
        v = y.view(bt // 8, 8, -1)                       #[b, t, chw]

        f = torch.matmul(q, k)
        f = F.softmax(f, dim=-1)                         #[b, t, t]

        out = torch.matmul(f, v)                         #[b, t, chw]
        out = out.view(-1, c, h, w)                      #[bt, c, h, w]     temporal-attention end
        out = out + y

        return  out

class try_att(nn.Module):
    def __init__(self, net):
        super(try_att, self).__init__()
        self.block = net
        self.ca_t = ChannelTemporal()
        self.sa_t = SpatialTemporal()


    def forward(self, x):
        x = self.block(x)
        x1 = self.ca_t(x) * x
        x2 = self.sa_t(x) * x

        x = x1 + x2
        return x

def NL3DWrapper(stage):
    blocks = list(stage.children())
    for i, b in enumerate(blocks):
        if i % 2 == 0:
            blocks[i].bn3 = try_att(b.bn3)
    return nn.Sequential(*blocks)


def make_non_local(net, n_segment):
    import torchvision

    if isinstance(net, torchvision.models.ResNet):
        net.layer1 = NL3DWrapper(net.layer1)
        net.layer2 = NL3DWrapper(net.layer2)
        net.layer3 = NL3DWrapper(net.layer3)
        net.layer4 = NL3DWrapper(net.layer4)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch 
    import torchvision.models as models
    model = models.resnet50(pretrained=False)

    input = torch.randn(8,3,224,224)
    make_non_local(model, n_segment=8)
    out = model(input)
    #print(model)
    for k, v in model.state_dict().items():
        print(k)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: %d" % pytorch_total_params)
