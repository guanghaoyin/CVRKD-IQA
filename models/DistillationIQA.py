import torch as torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
# from SemanticResNet50 import ResNetBackbone, Bottleneck
# from models.ResNet50_MLP import ResNetBackbone
# from models.MLP_return_inner_feature import MLPMixer

from functools import partial
from einops.layers.torch import Rearrange, Reduce

#ResNet
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class PatchEmbed(nn.Module):
    """ Feature to Patch Embedding
    input : N C H W
    output: N num_patch P^2*C
    """

    def __init__(self, patch_size=7, in_channels=2048):
        super().__init__()
        self.patch_size = patch_size
        self.dim = self.patch_size ** 2 * in_channels

    def forward(self, x):
        N, C, H, W = ori_shape = x.shape

        p = self.patch_size
        num_patches = (H // p) * (W // p)

        fold_out = torch.nn.functional.unfold(x, (p, p), stride=p) # B, num_dim, num_patch
        out = fold_out.permute(0, 2, 1 )# B, num_patch, num_dim

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetBackbone(nn.Module):

    # def __init__(self, outc=176, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1000, pretrained=True):
    def __init__(self, outc=2048, block=Bottleneck, layers=[3, 4, 6, 3], pretrained=True):
        super(ResNetBackbone, self).__init__()
        self.pretrained = pretrained
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.lda_out_channels = int(outc // 4)
        
        # local distortion aware module
        self.lda1_pool = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda1_fc = nn.Linear(16 * 64, self.lda_out_channels)

        self.lda2_pool = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda2_fc = nn.Linear(32 * 16, self.lda_out_channels)

        self.lda3_pool = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda3_fc = nn.Linear(64 * 4, self.lda_out_channels)

        self.lda4_pool = nn.AvgPool2d(7, stride=7)
        self.lda4_fc = nn.Linear(2048, self.lda_out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.pretrained:
            self.load_resnet50_backbone()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def cal_params(self):
        params = list(self.parameters())
        k = 0
        for i in params:
            l = 1
            for j in i.size():
                l *= j
            k = k + l
        print("Total parameters is :" + str(k))

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        # lda_1 = self.lda1_fc(self.lda1_pool(x).view(x.size(0), -1))
        lda_1 = x #[b, 256, 56, 56]
        x = self.layer2(x)
        # lda_2 = self.lda2_fc(self.lda2_pool(x).view(x.size(0), -1))
        lda_2 = x #[b, 512, 28, 28]
        x = self.layer3(x)
        # lda_3 = self.lda3_fc(self.lda3_pool(x).view(x.size(0), -1))
        lda_3 = x #[b, 1024, 14, 14]
        x = self.layer4(x)
        # lda_4 = self.lda4_fc(self.lda4_pool(x).view(x.size(0), -1))
        lda_4 = x #[b, 2048, 7, 7]
        # return x #[b, 2048, 7, 7]
        return [lda_1, lda_2, lda_3, lda_4]

    def load_resnet50_backbone(self):
        """Constructs a ResNet-50 model_hyper.
        Args:
            pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
        """
        save_model = model_zoo.load_url(model_urls['resnet50'])
        model_dict = self.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
    
def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    # if isinstance(m, nn.Conv2d):
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

#MLP
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class MLPMixer(nn.Module):
    def __init__(self, image_size, channels, patch_size, dim, depth, expansion_factor = 4, dropout = 0.):
        super().__init__()
        assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
        self.num_patches = (image_size // patch_size) ** 2
        self.chan_first, self.chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

        self.mlp = nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, self.FeedForward(self.num_patches, expansion_factor, dropout, self.chan_first)),
            PreNormResidual(dim, self.FeedForward(dim, expansion_factor, dropout, self.chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        # nn.Linear(dim, num_classes)
    )
        # print(self.mlp)

    def FeedForward(self, dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
        return nn.Sequential(
            dense(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            dense(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, distillation_layer_num=None):
        # [3, 256*self_patch_num, 7, 7]
        mlp_inner_feature = []
        layer_idx = 0
        for mlp_single in self.mlp:
            x = mlp_single(x)
            mlp_inner_feature.append(x)
        if distillation_layer_num:
            return x, mlp_inner_feature[-distillation_layer_num-2:-2]
        else:
            return x, mlp_inner_feature


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
            else:
                pass

class RegressionFCNet(nn.Module):
    """
    Target network for quality prediction.
    """
    def __init__(self):
        super(RegressionFCNet, self).__init__()
        self.target_in_size=512
        self.target_fc1_size=256

        self.sigmoid = nn.Sigmoid()
        self.l1 = nn.Linear(self.target_in_size, self.target_fc1_size)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(256)

        self.l2 = nn.Linear(self.target_fc1_size, 1)
        

    def forward(self, x):
        q = self.l1(x)
        q = self.l2(q).squeeze()
        return q

class DistillationIQANet(nn.Module):
    def __init__(self, self_patch_num=10, lda_channel=64, encode_decode_channel=64, MLP_depth=9, distillation_layer=9):
        super(DistillationIQANet, self).__init__()

        self.self_patch_num = self_patch_num
        self.lda_channel = lda_channel
        self.encode_decode_channel = encode_decode_channel
        self.MLP_depth = MLP_depth
        self.distillation_layer_num = distillation_layer

        self.feature_extractor = ResNetBackbone()
        for param in  self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.lda1_process = nn.Sequential(nn.Conv2d(256, self.lda_channel, kernel_size=1, stride=1, padding=0), nn.AdaptiveAvgPool2d((7, 7)))
        self.lda2_process = nn.Sequential(nn.Conv2d(512, self.lda_channel, kernel_size=1, stride=1, padding=0), nn.AdaptiveAvgPool2d((7, 7)))
        self.lda3_process = nn.Sequential(nn.Conv2d(1024, self.lda_channel, kernel_size=1, stride=1, padding=0), nn.AdaptiveAvgPool2d((7, 7)))
        self.lda4_process = nn.Sequential(nn.Conv2d(2048, self.lda_channel, kernel_size=1, stride=1, padding=0), nn.AdaptiveAvgPool2d((7, 7)))

        self.lda_process = [self.lda1_process, self.lda2_process, self.lda3_process, self.lda4_process]

        self.MLP_encoder_diff = MLPMixer(image_size = 7, channels = self.self_patch_num*self.lda_channel*4, patch_size = 1, dim = self.encode_decode_channel*4, depth = self.MLP_depth*2)
        self.MLP_encoder_lq = MLPMixer(image_size = 7, channels = self.self_patch_num*self.lda_channel*4, patch_size = 1, dim = self.encode_decode_channel*4, depth = self.MLP_depth)
        
        self.regressor = RegressionFCNet()

        initialize_weights(self.MLP_encoder_diff,0.1)
        initialize_weights(self.MLP_encoder_lq,0.1)
        initialize_weights(self.regressor,0.1)

        initialize_weights(self.lda1_process,0.1)
        initialize_weights(self.lda2_process,0.1)
        initialize_weights(self.lda3_process,0.1)
        initialize_weights(self.lda4_process,0.1)
    
    def forward(self, LQ_patches, refHQ_patches):
        device = LQ_patches.device
        b, p, c, h, w = LQ_patches.shape
        LQ_patches_reshape = LQ_patches.view(b*p, c, h, w)
        refHQ_patches_reshape = refHQ_patches.view(b*p, c, h, w)

        # [b*p, 256, 56, 56], [b*p, 512, 28, 28], [b*p, 1024, 14, 14], [b*p, 2048, 7, 7]
        lq_lda_features = self.feature_extractor(LQ_patches_reshape)
        refHQ_lda_features = self.feature_extractor(refHQ_patches_reshape)

        # encode_diff_feature, encode_lq_feature, feature = [], [], []
        multi_scale_diff_feature, multi_scale_lq_feature, feature = [], [], []
        for lq_lda_feature, refHQ_lda_feature, lda_process in zip(lq_lda_features, refHQ_lda_features, self.lda_process):
            # [b, p, 64, 7, 7]
            lq_lda_feature = lda_process(lq_lda_feature).view(b, -1, 7, 7)
            refHQ_lda_feature = lda_process(refHQ_lda_feature).view(b, -1, 7, 7)
            diff_lda_feature = refHQ_lda_feature - lq_lda_feature
            
            
            multi_scale_diff_feature.append(diff_lda_feature)
            multi_scale_lq_feature.append(lq_lda_feature)
           
        multi_scale_lq_feature = torch.cat(multi_scale_lq_feature, 1).to(device)
        multi_scale_diff_feature = torch.cat(multi_scale_diff_feature, 1).to(device)
        encode_lq_feature, encode_lq_inner_feature = self.MLP_encoder_lq(multi_scale_lq_feature, self.distillation_layer_num)
        encode_diff_feature, encode_diff_inner_feature = self.MLP_encoder_diff(multi_scale_diff_feature, self.distillation_layer_num)
        feature = torch.cat((encode_lq_feature, encode_diff_feature), 1)
        
        pred = self.regressor(feature)
        return encode_diff_inner_feature, encode_lq_inner_feature, pred
    
    def _load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))



if __name__ == "__main__":
    net = ResNetBackbone()
    x = torch.rand((1,3,224,224))
    y = net(x)
    print(y.shape)

    model = MLPMixer(image_size = 7, channels = 1280, patch_size = 1, dim = 512, depth = 12)
    img = torch.randn(96, 256, 7, 7)
    pred = model(img) # (1, 1000)  
    print(pred.shape) 

    m = DistillationIQANet()
    lq = torch.rand((3,10,3,224,224))
    hq = torch.rand((3,10,3,224,224))
    encode_diff_feature, encode_lq_feature, pred = m(lq, hq)
    print(pred.shape)
