
# resent18
# @article{isaksen2025raw,
#   title={Raw photoplethysmogram waveforms versus peak-to-peak intervals for machine learning detection of atrial fibrillation: Does waveform matter?},
#   author={Isaksen, Jonas L and Arildsen, Bolette and Lind, Cathrine and N{\o}rregaard, Malene and Vernooy, Kevin and Schotten, Ulrich and Jespersen, Thomas and Betz, Konstanze and Hermans, Astrid NL and Kanters, J{\o}rgen K and others},
#   journal={Computer Methods and Programs in Biomedicine},
#   volume={260},
#   pages={108537},
#   year={2025},
#   publisher={Elsevier}
# }
# https://www.sciencedirect.com/science/article/pii/S0169260724005303

from config import params
import torch
import torch.nn as nn

# Basic ResNet1D block
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

# Bottleneck ResNet1D block
class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

# ResNet1D structure
class ResNet1D(nn.Module):
    def __init__(self, block, layers, input_dim=1, num_classes=4):
        super(ResNet1D, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Input channel selection
def get_input_dim_from_params():
    use_channel = params.get('use_channel', 'ppg_acc')
    dual_channel = params.get('dual_channel', False)

    if use_channel == 'ppg':
        return 2 if dual_channel else 1
    elif use_channel == 'acc':
        return 3
    else:  # 'ppg_acc'
        return 5 if dual_channel else 4

# ResNet1D constructors
def resnet1D18(num_classes=3):
    input_dim = get_input_dim_from_params()
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], input_dim=input_dim, num_classes=num_classes)

def resnet1D34(num_classes=3):
    input_dim = get_input_dim_from_params()
    return ResNet1D(BasicBlock1D, [3, 4, 6, 3], input_dim=input_dim, num_classes=num_classes)

def resnet1D50(num_classes=3):
    input_dim = get_input_dim_from_params()
    return ResNet1D(Bottleneck1D, [3, 4, 6, 3], input_dim=input_dim, num_classes=num_classes)

def resnet1D101(num_classes=3):
    input_dim = get_input_dim_from_params()
    return ResNet1D(Bottleneck1D, [3, 4, 23, 3], input_dim=input_dim, num_classes=num_classes)
