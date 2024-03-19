import torch
import torch.nn as nn
from collections import OrderedDict

class f_common(nn.Module):
    def __init__(self):
        self.padding_edf = {  # same padding in tensorflow
            'conv1': (22, 22),
            'max_pool1': (2, 2),
            'conv2': (3, 4),
            'max_pool2': (0, 1),
        }
        super(f_common,self).__init__()
        self.cnn = nn.Sequential(nn.ConstantPad1d(self.padding_edf['conv1'], 0),
                                 nn.Sequential(OrderedDict([('conv1_1',
                                     nn.Conv1d(in_channels=1, out_channels=128, kernel_size=50, stride=6,
                                               bias=False))])),
                                 nn.ReLU(inplace=True),
                                 nn.ConstantPad1d(self.padding_edf['max_pool1'], 0),
                                 nn.MaxPool1d(8, 8),
                                 nn.Dropout(p=0.5),
                                 nn.ConstantPad1d(self.padding_edf['conv2'], 0),
                                 nn.Sequential(OrderedDict([('conv2_2',
                                     nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1,
                                               bias=False))])),
                                 nn.ReLU(inplace=True))

    def forward(self,x):
        x = self.cnn(x)
        return x

class f_domain(nn.Module):
    def __init__(self):
        self.padding_edf = {  # same padding in tensorflow
            'conv1': (22, 22),
            'max_pool1': (2, 2),
            'conv2': (3, 4),
            'max_pool2': (0, 1),
        }
        super(f_domain,self).__init__()
        self.cnn = nn.Sequential(nn.ConstantPad1d(self.padding_edf['conv2'], 0),
                                 nn.Sequential(OrderedDict([('conv2_1',nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1,bias=False))])),
                                 nn.ReLU(inplace=True),
                                 nn.ConstantPad1d(self.padding_edf['conv2'], 0),
                                 nn.Sequential(OrderedDict([('conv2_2',
                                     nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1,
                                               bias=False))])),
                                 nn.ReLU(inplace=True),
                                 nn.ConstantPad1d(self.padding_edf['max_pool2'], 0),
                                 nn.MaxPool1d(4,4),
                                 )
        self.ave = nn.AvgPool1d(16,16)
        self.max = nn.MaxPool1d(16, 16)
        self.dp = nn.Dropout(p=0.5)

    def forward(self,x):
        x = self.cnn(x)
        ave = self.ave(x)
        max = self.max(x)
        x = torch.cat([ave,max],1)
        x = x.squeeze(2)
        x = self.dp(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_domains=2):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains),
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Export_Gap(nn.Module):
    def __init__(self):
        super(Export_Gap, self).__init__()
        self.avegap = nn.AdaptiveAvgPool1d(5)
        self.maxgap = nn.AdaptiveMaxPool1d(5)
        self.export = nn.Sequential(nn.Linear(20, 2),
                                    nn.Dropout(p=0.2),
                                    nn.Softmax(dim=1))

    def forward(self, feature_src_mix,feature_src):
        ave_mix = self.avegap(feature_src_mix.unsqueeze(dim=1)).squeeze(dim=1)
        ave_s = self.avegap(feature_src.unsqueeze(dim=1)).squeeze(dim=1)
        max_mix = self.maxgap(feature_src_mix.unsqueeze(dim=1)).squeeze(dim=1)
        max_s = self.maxgap(feature_src.unsqueeze(dim=1)).squeeze(dim=1)
        pred_weight = self.export(torch.cat((ave_mix,max_mix,ave_s,max_s), dim=1))
        return pred_weight

