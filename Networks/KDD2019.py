# @inproceedings{10.1145/3292500.3330657,
# author = {Shen, Yichen and Voisin, Maxime and Aliamiri, Alireza and Avati, Anand and Hannun, Awni and Ng, Andrew},
# title = {Ambulatory Atrial Fibrillation Monitoring Using Wearable Photoplethysmography with Deep Learning},
# year = {2019},
# isbn = {9781450362016},
# publisher = {Association for Computing Machinery},
# address = {New York, NY, USA},
# url = {https://doi.org/10.1145/3292500.3330657},
# doi = {10.1145/3292500.3330657},
# abstract = {We develop an algorithm that accurately detects Atrial Fibrillation (AF) episodes from photoplethysmograms (PPG) recorded in ambulatory free-living conditions. We collect and annotate a dataset containing more than 4000 hours of PPG recorded from a wrist-worn device. Using a 50-layer convolutional neural network, we achieve a test AUC of 95\% in presence of motion artifacts inherent to PPG signals. Such continuous and accurate detection of AF has the potential to transform consumer wearable devices into clinically useful medical monitoring tools.},
# booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
# pages = {1909–1916},
# numpages = {8},
# keywords = {ppg, deep learning, convolutional neural network, atrial fibrillation, ambulatory},
# location = {Anchorage, AK, USA},
# series = {KDD '19}
# }

# https://dl.acm.org/doi/10.1145/3292500.3330657

from config import params
import torch
import torch.nn as nn
# ResNeXt1D model based on the KDD 2019 design
import torch
import torch.nn as nn
import torch.nn.functional as F

# Bottleneck block with grouped convolution
class ResNeXtBottleneck1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=4):
        super().__init__()
        group_width = cardinality * base_width

        self.conv_reduce = nn.Conv1d(in_channels, group_width, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm1d(group_width)

        self.conv_conv = nn.Conv1d(group_width, group_width, kernel_size=3, stride=stride, padding=1, 
                                   groups=cardinality, bias=False)
        self.bn = nn.BatchNorm1d(group_width)

        self.conv_expand = nn.Conv1d(group_width, out_channels, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn_reduce(self.conv_reduce(x)))
        out = self.relu(self.bn(self.conv_conv(out)))
        out = self.bn_expand(self.conv_expand(out))
        out += self.shortcut(x)
        return self.relu(out)


# ResNeXt1D model
class ResNeXt1D(nn.Module):
    def __init__(self, input_channels=1, num_classes=3, cardinality=32, base_width=4, layers=[3, 4, 6, 3]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        channels = [128, 256, 512, 1024]
        self.stage1 = self._make_layer(64, channels[0], layers[0], stride=1, cardinality=cardinality, base_width=base_width)
        self.stage2 = self._make_layer(channels[0], channels[1], layers[1], stride=2, cardinality=cardinality, base_width=base_width)
        self.stage3 = self._make_layer(channels[1], channels[2], layers[2], stride=2, cardinality=cardinality, base_width=base_width)
        self.stage4 = self._make_layer(channels[2], channels[3], layers[3], stride=2, cardinality=cardinality, base_width=base_width)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[3], num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride, cardinality, base_width):
        layers = [ResNeXtBottleneck1D(in_channels, out_channels, stride, cardinality, base_width)]
        for _ in range(1, blocks):
            layers.append(ResNeXtBottleneck1D(out_channels, out_channels, 1, cardinality, base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)


# Factory function function
def KDD2019(num_classes=3):
    from config import params
    use_channel = params.get('use_channel', 'ppg_acc')
    dual_channel = params.get('dual_channel', False)

    if use_channel == 'ppg':
        in_channels = 2 if dual_channel else 1
    elif use_channel == 'acc':
        in_channels = 3
    else:  # 'ppg_acc'
        in_channels = 5 if dual_channel else 4

    return ResNeXt1D(input_channels=in_channels, num_classes=num_classes)