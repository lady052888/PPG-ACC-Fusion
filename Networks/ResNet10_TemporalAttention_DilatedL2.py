# proposed model in this paper

from config import params, get_num_classes
import torch
import torch.nn as nn
import torch.nn.functional as F

# Temporal attention module
class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super(TemporalAttention, self).__init__()
        self.query_conv = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (B, C, T)
        Q = self.query_conv(x).permute(0, 2, 1)        # (B, T, C_q)
        K = self.key_conv(x)                           # (B, C_q, T)
        V = self.value_conv(x)                         # (B, C, T)

        attention = self.softmax(torch.bmm(Q, K))      # (B, T, T)
        out = torch.bmm(V, attention.permute(0, 2, 1)) # (B, C, T)
        return out + x  # residual connection


# Basic residual block
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, p=0.5):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.dropout1 = nn.Dropout(p)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.dropout2 = nn.Dropout(p)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DilatedBasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, p=0.5, dilation=2):
        super(DilatedBasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.dropout1 = nn.Dropout(p)
        self.conv2 = nn.Conv1d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.dropout2 = nn.Dropout(p)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ResNet1D with temporal attention
class ResNet1D_DilatedL2SecondBlock(nn.Module):
    def __init__(
        self,
        num_classes=4,
        p=0.5,
        in_channels=1,
        dilation=2
    ):
        super(ResNet1D_DilatedL2SecondBlock, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(p)

        self.layer1 = self._make_layer(BasicBlock1D, 32, 2, stride=1, p=p)

        layer2_blocks = []
        layer2_blocks.append(
            BasicBlock1D(self.in_planes, 64, stride=2, p=p)
        )
        self.in_planes = 64
        layer2_blocks.append(
            DilatedBasicBlock1D(self.in_planes, 64, stride=1, p=p, dilation=dilation)
        )
        self.layer2 = nn.Sequential(*layer2_blocks)

        # === Temporal Attention after layer2 ===
        self.temporal_attn = TemporalAttention(64)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, p):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, p))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.temporal_attn(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNet10_TemporalAttention_DilatedL2(num_classes=None, p=0.2, dilation=2):
    if num_classes is None:
        num_classes = get_num_classes()
    use_channel = params.get('use_channel', 'ppg_acc')
    dual_channel = params.get('dual_channel', False)

    use_channel = str(use_channel).lower()
    if use_channel == 'ppg':
        in_channels = 2 if dual_channel else 1
    elif use_channel == 'acc':
        in_channels = 3
    elif use_channel == 'ppg_acc':
        in_channels = 5 if dual_channel else 4
    elif use_channel in ('4ch', 'ppg_hr_accmag_maghr'):
        in_channels = 4
        if dual_channel:
            pass
        assert not params.get('apply_autocorrelation', False), \
            "In '4ch' mode, set params['apply_autocorrelation']=False."
    else:
        raise ValueError(f"Unknown use_channel: {use_channel}")

    return ResNet1D_DilatedL2SecondBlock(
        num_classes=num_classes,
        p=p,
        in_channels=in_channels,
        dilation=dilation
    )
