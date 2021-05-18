# Non-local block using embedded gaussian
import torch
from torch import nn
from torch.nn import functional as F

def _to_4d_tensor(x):
    x = x.transpose(0, 2)  # swap batch and depth dimensions: NxCxDxHxW => DxCxNxHxW
    x = x.permute(2, 0, 1, 3, 4)  # DxCxNxHxW => NxDxCxHxW
    x = torch.split(x, 1, dim=0)  # split along batch dimension: NxDxCxHxW => N*[1xDxCxHxW]
    x = torch.cat(x, 1)  # concatenate along depth dimension: N*[1xDxCxHxW] => 1x(N*D)xCxHxW
    x = x.squeeze(0)  # 1x(N*D)xCxHxW => (N*D)xCxHxW
    return x

class try_shuffle(nn.Module):
    def __init__(self, block, n_segment):
        super(try_shuffle, self).__init__()
        self.block = block
        self.n_segment = n_segment
        self.shuffle_conv1 = nn.Conv2d(n_segment, n_segment, kernel_size=3, stride=1, padding=1)
#        self.shuffle_bn = nn.BatchNorm2d(n_segment, affine=True)
#        self.shuffle_relu = nn.ReLU (inplace=True)

    def forward(self, x):

        x1 = self.block(x)

        bd, c, h, w = x.size()
        x = x.view(bd // self.n_segment, self.n_segment, c, h, w)  #[b, d, c, h, w]
        n_b, n_c, n_d, n_h, n_w = x.size()

        x = _to_4d_tensor(x)                              #[bc, d, h, w]
        x = self.shuffle_conv1(x)                         #[bc, d, h, w]
#        x = self.shuffle_bn(x)                            #[bc, d, h, w]
#        x = self.shuffle_relu(x)                          #[bc, d, h, w]

        x = torch.split(x, n_d)  # (N*D)xCxHxW => N*[DxCxHxW]
        x = torch.stack(x, dim=0)  # re-instate the batch dimension: NxDxCxHxW
        x = x.transpose(1, 2)  # swap back depth and channel dimensions: NxDxCxHxW => NxCxDxHxW
        x = x.permute(0, 2, 1, 3, 4)           #[b, c, d, h, w]

        x = _to_4d_tensor(x)                   #[bd, c, h, w]

        x2 = self.block(x)
        
        x =x1 + x2
        return x


def make_stage(stage, n_segment):
    blocks = list(stage.children())
    for i, b in enumerate(blocks):
        if i % 2 == 0:
            blocks[i].conv2 = try_shuffle(b.conv2, n_segment)
    return nn.Sequential(*blocks)


def make_shuffle_c_t(net, n_segment):
    import torchvision

    if isinstance(net, torchvision.models.ResNet):
        net.layer2 = make_stage(net.layer2, n_segment)
        net.layer3 = make_stage(net.layer3, n_segment)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch 
    import torchvision.models as models
    from torchsummary import summary
    model = models.resnet50(pretrained=False)

    input = torch.randn(8,3,224,224)
    make_shuffle_c_t(model, n_segment=8)
    out = model(input)
    print("test is over")
#    summary(model, input)
    print(model)
    for k, v in model.state_dict().items():
        print(k)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: %d" % pytorch_total_params)
