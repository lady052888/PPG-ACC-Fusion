from config import params
import torch
import torch.nn as nn

# Liu Z, Zhou B, Jiang Z, Chen X, Li Y, Tang M, Miao F. Multiclass Arrhythmia Detection and Classification From 
# Photoplethysmography Signals Using a Deep Convolutional Neural Network. J Am Heart Assoc. 2022 Apr 5;11(7):e023555. 
# doi: 10.1161/JAHA.121.023555. Epub 2022 Mar 24. PMID: 35322685; PMCID: PMC9075456.

# Basic block: Conv1d + BN + ReLU
class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class VGG16_1D_JAHA(nn.Module):
    """
    1D adaptation of the JAHA paper:
    'Multiclass Arrhythmia Detection and Classification From
     Photoplethysmography Signals Using a Deep Convolutional
     Neural Network'.

    Architecture:
      - 13 Conv1d layers (32,32,64,64,128,128,128,256,256,256,256,256,256)
      - Each Conv followed by BN + ReLU
      - 5 MaxPool1d (kernel_size=3, stride=3)
      - Adaptive pooling to length 4
      - 2 FC layers: 1024 -> 256 -> num_classes
    """

    def __init__(self, in_channels: int, num_classes: int = 3):
        super().__init__()

        # Block1: output 32, length unchanged
        self.block1 = nn.Sequential(
            ConvBNReLU1D(in_channels, 32, kernel_size=3, stride=1, padding=1),
            ConvBNReLU1D(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
        )
        # expected: BS×32×1000 -> BS×32×333

        # Block2: output 64
        self.block2 = nn.Sequential(
            ConvBNReLU1D(32, 64, kernel_size=3, stride=1, padding=1),
            ConvBNReLU1D(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
        )
        # expected: BS×64×333 -> BS×64×111

        # Block3: output 128
        self.block3 = nn.Sequential(
            ConvBNReLU1D(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBNReLU1D(128, 128, kernel_size=3, stride=1, padding=1),
            ConvBNReLU1D(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
        )
        # expected: BS×128×111 -> BS×128×37

        # Block4: output 256
        self.block4 = nn.Sequential(
            ConvBNReLU1D(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBNReLU1D(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBNReLU1D(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
        )
        # expected: BS×256×37 -> BS×256×12

        # Block5: still 256
        self.block5 = nn.Sequential(
            ConvBNReLU1D(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBNReLU1D(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBNReLU1D(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
        )
        # expected: BS×256×12 -> BS×256×4

        # Adaptive pool to length 4 for arbitrary input lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(4)  # output BS×256×4

        # Fully connected: 256*4 -> 256 -> num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # BS×(256*4)
            nn.Linear(256 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        x: (batch_size, in_channels, seq_len)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        # Adaptive pool in case input length differs from 1000
        x = self.adaptive_pool(x)  # (B, 256, 4)

        x = self.classifier(x)     # (B, num_classes)
        return x


# Factory function function
def VGG16_1D_Multimodal(num_classes: int = 3):
   
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
            # dual_channel ignored in 4ch mode
            pass
        assert not params.get('apply_autocorrelation', False), \
            "In '4ch' mode, set params['apply_autocorrelation']=False."

    else:
        raise ValueError(f"Unknown use_channel: {use_channel}")

    model = VGG16_1D_JAHA(
        in_channels=in_channels,
        num_classes=num_classes
    )
    return model


