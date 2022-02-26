import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

class DCNN_NARIQA(nn.Module):
    def __init__(self):
        super().__init__()
        #ref path
        self.block1_ref = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=1))
        self.block2_ref = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=9, stride=1),
            nn.ReLU())
        self.fc3_ref = nn.Linear(in_features=59168, out_features=1024)

        #LQ path
        self.block1_lq = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=1))
        self.block2_lq = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=9, stride=1),
            nn.ReLU())
        self.fc3_lq = nn.Linear(in_features=59168, out_features=1024)

        self.fc4 = nn.Linear(in_features=2048, out_features=1024)
        self.fc5 = nn.Linear(in_features=1024, out_features=1)
    
    def forward(self, lq_patches, ref_patches):
        feature_lq = self.block1_lq(lq_patches)
        feature_lq = self.block2_lq(feature_lq)
        feature_lq = self.fc3_lq(feature_lq.view(feature_lq.size(0), -1))

        feature_ref = self.block1_ref(ref_patches)
        feature_ref = self.block2_ref(feature_ref)
        feature_ref = self.fc3_ref(feature_ref.view(feature_ref.size(0), -1))

        concat_feature = torch.cat((feature_ref, feature_lq), 1)
        concat_feature = self.fc4(concat_feature)
        pred = self.fc5(concat_feature)
        return pred

if __name__ == "__main__":
    x = torch.rand((1,3,224,224))
    y = torch.rand((1,3,224,224))
    net = DCNN_NARIQA()
    pred = net(x, y)
