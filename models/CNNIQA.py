import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

class CNNIQAnet(nn.Module):
    def __init__(self):
        super(CNNIQAnet, self).__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=7)
        self.fc1 = nn.Linear(100, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 1)

    def forward(self, input):
        x = input.view(-1, input.size(-3), input.size(-2), input.size(-1))

        x = self.conv(x)

        x1 = F.max_pool2d(x, (x.size(-2), x.size(-1)))
        x2 = -F.max_pool2d(-x, (x.size(-2), x.size(-1)))

        h = torch.cat((x1, x2), 1)
        h = h.squeeze(3).squeeze(2)

        h = F.relu(self.fc1(h))
        h = F.dropout(h)
        h = F.relu(self.fc2(h))

        q = self.fc3(h)

        return q
