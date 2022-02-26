import torch as torch
from torch._C import device
import torch.nn as nn
import math
from torch.nn.modules.conv import Conv2d
import torch.utils.model_zoo as model_zoo
from torch.nn import Dropout, Softmax, Linear, LayerNorm
# from SemanticResNet50 import ResNetBackbone, Bottleneck
# from models.ResNet50_MLP import ResNetBackbone
# from models.MLP_return_inner_feature import MLPMixer


#ResNet
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

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


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

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

class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.hidden_size = 32
        self.mlp_size = 64
        self.dropout_rate = 0.1
        self.fc1 = Linear(self.hidden_size, self.mlp_size)
        self.fc2 = Linear(self.mlp_size, self.hidden_size)
        self.act_fn = nn.ReLU()
        self.dropout = Dropout(self.dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MHA(nn.Module):
    def __init__(self):
        super(MHA, self).__init__()
        self.vis = True
        self.num_attention_heads = 8
        self.hidden_size = 32
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(self.hidden_size, self.all_head_size)
        self.key = Linear(self.hidden_size, self.all_head_size)
        self.value = Linear(self.hidden_size, self.all_head_size)

        self.out = Linear(self.hidden_size, self.hidden_size)
        self.attn_dropout = Dropout(0.0)
        self.proj_dropout = Dropout(0.0)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        mixed_key_layer = self.key(x)
        mixed_value_layer = self.value(x)
        mixed_query_layer = self.query(x)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output

class TransformerBlock(nn.Module):
    def __init__(self):
        super(TransformerBlock, self).__init__()
        self.hidden_size = 32
        self.ffn_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn = Mlp()
        self.attention_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.attn = MHA()
        
        self.dropout_rate = 0.1
        self.dropout1 = Dropout(self.dropout_rate)
        self.dropout2 = Dropout(self.dropout_rate)

    def forward(self, x):
        x1 = self.attn(x)
        x1 = self.dropout1(x1)
        x1 += x
        x1 = self.attention_norm(x1)

        x2 = self.ffn(x1)
        x2 = self.dropout2(x2)
        x2 += x1
        x2 = self.attention_norm(x2)

        return x2

class RegressionFCNet(nn.Module):
    def __init__(self):
        super(RegressionFCNet, self).__init__()
        self.target_in_size=32
        self.target_fc1_size=64

        self.l1 = nn.Linear(self.target_in_size, self.target_fc1_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(self.target_fc1_size, 1)
        

    def forward(self, x):
        q = self.l1(x)
        q = self.relu(q)
        q = self.l2(q).squeeze()
        return q

class TRIQ(nn.Module):
    def __init__(self):
        super(TRIQ, self).__init__()
        self.feature_extractor = ResNetBackbone()
        # for param in  self.feature_extractor.parameters():
            # param.requires_grad = False
        
        self.conv = Conv2d(2048,32,kernel_size=1, stride=1, padding=0)
        self.position_embeddings = nn.Parameter(torch.zeros(1, 49+1, 32))
        self.quality_token = nn.Parameter(torch.zeros(1, 1, 32))

        self.transformer_block1 = TransformerBlock()
        self.transformer_block2 = TransformerBlock()

        self.regressor = RegressionFCNet()
    
    def cal_params(self):
        params = list(self.parameters())
        k = 0
        for i in params:
            l = 1
            for j in i.size():
                l *= j
            k = k + l
        print("Total parameters is :" + str(k))

    def forward(self, LQ):
        B = LQ.shape[0]
        feature_LQ = self.feature_extractor(LQ)
        feature_LQ = self.conv(feature_LQ)

        quality_tokens = self.quality_token.expand(B,-1,-1)

        flat_feature_LQ = feature_LQ.flatten(2).transpose(-1,-2)
        flat_feature_LQ = torch.cat((quality_tokens,flat_feature_LQ), dim=1) + self.position_embeddings

        f = self.transformer_block1(flat_feature_LQ)
        f = self.transformer_block2(f)

        y = self.regressor(f)

        return y

'''
TEST
Run this code with:
```
cd $HOME/pretrained-models.pytorch
python -m pretrainedmodels.inceptionresnetv2
```
'''
if __name__ == '__main__':
    import time
    device = torch.device('cuda')
    LQ = torch.rand((1,3,224, 224)).to(device)
    net = TRIQ().to(device)
    net.cal_params()

    torch.cuda.synchronize()
    start = time.time()
    a = net(LQ)
    torch.cuda.synchronize()
    end = time.time()
    print("run time is :" + str(end-start))
