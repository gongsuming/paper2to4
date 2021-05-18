# Non-local block using embedded gaussian
import torch
from torch import nn
from torch.nn import functional as F


class try_i4(nn.Module):
    def __init__(self, block, n_segment, channel):
        super(try_i4, self).__init__()
        self.block = block
        self.n_segment = n_segment
        self.channel = channel
        self.q = nn.Parameter(torch.ones(1))
        self.k = nn.Parameter(torch.ones(1))
        #self.v = nn.Parameter(torch.ones(1))
        #self.conv2d_00 = nn.Conv2d( self.channel * 3, self.channel, (1, 1), stride=1, bias=False)
        self.conv2d_0 = nn.Conv2d( 1, 1 , (3, 3), stride=1, padding=(1, 1),dilation=(1,1), bias=False)
        self.conv2d_1 = nn.Conv2d( 1, 1 , (3, 3), stride=1, padding=(2, 2),dilation=(2,2), bias=False)
        #self.conv2d_2 = nn.Conv2d( 1, 1 , (3, 3), stride=1, padding=(3, 3),dilation=(3,3), bias=False)
        self.r = nn.ReLU(inplace=True)

    def forward(self, x):

        x2, x3 = self.sample_i4(x, self.n_segment)
        x = x + x2 + x3 #+ x4
        #x = torch.cat([x, x2, x3], 1)
        #x = self.conv2d_00(x)
        #x = self.r(x)
        x = self.block(x)

        return x

    #@staticmethod
    def sample_i4(self, x, n_segment):
        nt, c, h, w = x.data.size()
        n_batch = nt // n_segment
        x1 = x.view(n_batch, n_segment, c, h, w)              # [ b, t, c, h, w]
        x1 = x1.view(n_batch, n_segment,-1).unsqueeze(1)      # [ b, 1, t, chw]
        x2 = self.conv2d_0(x1) * self.q
        x2 = self.r(x2)
        x3 = self.conv2d_1(x1) * self.k
        x3 = self.r(x3)
        #x4 = self.conv2d_2(x1) * self.v
        #x4 = self.r(x4)

        x2 = x2.squeeze(1).view(-1, c, h, w)
        x3 = x3.squeeze(1).view(-1, c, h, w)
        #x4 = x4.squeeze(1).view(-1, c, h, w)

        return x2, x3#, x4




def make_stage(stage, n_segment, channel):
    blocks = list(stage.children())
    for i, b in enumerate(blocks):
        if i>= 0:
            blocks[i].conv2 = try_i4(b.conv2, n_segment, channel)
    return nn.Sequential(*blocks)


def try_sample_i4(net, n_segment):
    import torchvision

    if isinstance(net, torchvision.models.ResNet):
        #net.layer1 = make_stage(net.layer1, n_segment)
        #net.layer2 = make_stage(net.layer2, n_segment, channel = 128)
        #net.layer3 = make_stage(net.layer3, n_segment, channel = 256)
        net.layer4 = make_stage(net.layer4, n_segment, channel = 512)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch 
    import torchvision.models as models

    model = models.resnet50(pretrained=False)

    input = torch.randn(8,3,224,224)
    try_sample_i4(model, n_segment=8)
    out = model(input)
    print("test is over")

    #print(model)
    for k, v in model.state_dict().items():
        print(k)
    #pytorch_total_params = sum(p.numel() for p in model.parameters())
    #print("Total number of parameters: %d" % pytorch_total_params)

    from thop import profile

    flops, params = profile(model, inputs=(input, ))
    print(flops, params)    
