from config import params
import torch
import torch.nn as nn

# Depthwise separable 1D convolution
class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        return self.relu(out)

class MobileNet1D(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            DepthwiseSeparableConv1D(32, 64),
            DepthwiseSeparableConv1D(64, 128),
            DepthwiseSeparableConv1D(128, 128),
            DepthwiseSeparableConv1D(128, 256),
            DepthwiseSeparableConv1D(256, 256),
            DepthwiseSeparableConv1D(256, 512),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):  # x: (batch, input_dim, seq_len)
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Channel adapter
def get_input_dim_from_params():
    use_channel = params.get('use_channel', 'ppg_acc')
    dual_channel = params.get('dual_channel', False)

    if use_channel == 'ppg':
        return 2 if dual_channel else 1
    elif use_channel == 'acc':
        return 3
    else:  # 'ppg_acc'
        return 5 if dual_channel else 4

# Factory function function
def mobilenet_1d(num_classes=3):
    input_dim = get_input_dim_from_params()
    return MobileNet1D(input_dim=input_dim, num_classes=num_classes)
