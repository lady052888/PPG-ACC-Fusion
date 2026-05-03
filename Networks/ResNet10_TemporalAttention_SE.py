# ResNet10 with Temporal Attention and SE blocks (paper: Zhao25-RhythmiNet)
# @inproceedings{zhao2025motion,
#   title={Motion-Robust Multimodal Fusion of PPG and Accelerometer Signals for Three-Class Heart Rhythm Classification},
#   author={Zhao, Yangyang and Kaisti, Matti and Lahdenoja, Olli and Koivisto, Tero},
#   booktitle={Companion of the 2025 ACM International Joint Conference on Pervasive and Ubiquitous Computing},
#   pages={171--175},
#   year={2025}
# }

from config import params, get_num_classes
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super(TemporalAttention, self).__init__()
        self.query_conv = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query_conv(x).permute(0, 2, 1)
        K = self.key_conv(x)
        V = self.value_conv(x)
        attention = self.softmax(torch.bmm(Q, K))
        out = torch.bmm(V, attention.permute(0, 2, 1))
        return out + x


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _ = x.size()
        y = self.global_avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1)
        return x * y


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, p=0.5, use_se=False, reduction=16):
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

        self.use_se = use_se
        if self.use_se:
            self.se_block = SEBlock(planes, reduction)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        if self.use_se:
            out = self.se_block(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4, p=0.5, use_se=False, reduction=16, in_channels=1):
        super(ResNet1D, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(p)

        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1, p=p, use_se=use_se, reduction=reduction)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2, p=p, use_se=use_se, reduction=reduction)

        self.temporal_attn = TemporalAttention(64)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, p, use_se, reduction):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, p, use_se, reduction))
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


def ResNet10_TemporalAttention_SE(num_classes=None, p=0.2, use_se=True, reduction=8):
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

    return ResNet1D(
        block=BasicBlock1D,
        num_blocks=[2, 2],
        num_classes=num_classes,
        p=p,
        use_se=use_se,
        reduction=reduction,
        in_channels=in_channels
    )
