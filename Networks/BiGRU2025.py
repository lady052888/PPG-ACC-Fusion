# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11176804
# @ARTICLE{11176804,
#   author={Han, Dong and Moon, Jihye and Díaz, Luís R. Mercado and Chen, Darren and Williams, Devan and Mohagheghian, Fahimeh and Ghetia, Om and Peitzsch, Andrew G. and Kong, Youngsun and Nishita, Nishat and Ghutadaria, Ohm and Orwig, Taylor A. and Otabil, Edith Mensah and Noorishirazi, Kamran and Hamel, Alexander and Dickson, Emily L. and DiMezza, Danielle and Lessard, Darleen and Wang, Ziyue and Paul, Tenes and Mehawej, Jordy and Filippaios, Andreas and Naeem, Syed and Gottbrecht, Matthew F. and Fitzgibbons, Timothy P. and Saczynski, Jane S. and Barton, Bruce and Ding, Eric Y. and Tran, Khanh-Van and McManus, David D. and Chon, Ki H.},
#   journal={IEEE Transactions on Biomedical Engineering}, 
#   title={Multiclass Arrhythmia Classification using Multimodal Smartwatch Photoplethysmography Signals Collected in Real-life Settings}, 
#   year={2025},
#   volume={},
#   number={},
#   pages={1-14},
#   keywords={Electrocardiography;Wearable Health Monitoring Systems;Recording;Clinical trials;Accuracy;Computational modeling;Arrhythmia;Monitoring;Data models;Rhythm;Atrial Fibrillation;Premature Atrial Contraction;Premature Ventricular Contraction;Wearable Device;Photoplethysmography;Clinical Trial;Deep Learning},
#   doi={10.1109/TBME.2025.3613471}}

from config import params
import torch
import torch.nn as nn


class BiGRUArrhythmiaNet(nn.Module):
    """
    Reimplementation of the 1D-Bi-GRU architecture in
    'Multiclass Arrhythmia Classification using Multimodal Smartwatch PPG Signals'
    (network definition only, no training logic).

    Input:
        x: (batch_size, in_channels, seq_len)

    Output:
        logits: (batch_size, num_classes)           segment-level output (mean over time)
        logits_seq: (batch_size, seq_len, num_classes) per-timestep output
    """

    def __init__(self, in_channels: int, num_classes: int = 3,
                 hidden_size: int = 128, dropout_p: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # 1D CNN: (B, d, L) -> (B, 4d, L)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=4 * in_channels,   # 4d filters
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False
        )

        # Bi-GRU: input feature dim = 4d, hidden_size=128, single-layer bidirectional
        self.gru = nn.GRU(
            input_size=4 * in_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,   # input (B, L, C)
            bidirectional=True
        )

        # Bi-GRU output feature dim = 2 * hidden_size
        self.bn = nn.BatchNorm1d(2 * hidden_size)
        self.dropout = nn.Dropout(p=dropout_p)

        # Fully connected: 2H -> num_classes(3)
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x):
        """
        x: (batch_size, in_channels, seq_len)
        """
        # 1D CNN: (B, d, L) -> (B, 4d, L)
        x = self.conv(x)  # no extra ReLU, following the original description

        # Permute for GRU: (B, 4d, L) -> (B, L, 4d)
        x = x.permute(0, 2, 1)

        # Bi-GRU: (B, L, 4d) -> (B, L, 2H)
        x, _ = self.gru(x)

        # BatchNorm works on channel dimension, needs (B, C, L)
        x = x.permute(0, 2, 1)      # (B, 2H, L)
        x = self.bn(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)      # (B, L, 2H)

        # Per-timestep classification: (B, L, 2H) -> (B, L, num_classes)
        logits_seq = self.fc(x)

        # Segment-level output: mean over time
        logits = logits_seq.mean(dim=1)   # (B, num_classes)

        return logits, logits_seq


# Factory function function
def BiGRU_Multimodal(num_classes: int = 3,
                     p: float = 0.2,
                     hidden_size: int = 128):
    """
    Same as KDD2019(): read use_channel / dual_channel from config.params
    to determine in_channels.

    Current convention:
      - use_channel == 'ppg':
          in_channels = 1 (single-channel PPG)
          if dual_channel=True: in_channels = 2 (PPG + autocorr)

      - use_channel == 'acc':
          in_channels = 3 (ACC x,y,z)

      - use_channel == 'ppg_acc':
          in_channels = 4 (PPG + ACC x,y,z)
          if dual_channel=True: in_channels = 5 (adds PPG autocorr)

      - use_channel in ['4ch', 'ppg_hr_accmag_maghr']:
          in_channels = 4, corresponds to the paper's best model:
              [PPG, HR_line_32Hz, acc_mag, magHR_32Hz]
          dual_channel and autocorr are forced off here
    """
    use_channel = params.get('use_channel', 'ppg_acc')
    dual_channel = params.get('dual_channel', False)

    use_channel = str(use_channel).lower()

    if use_channel == 'ppg':
        in_channels = 2 if dual_channel else 1

    elif use_channel == 'acc':
        in_channels = 3

    elif use_channel == 'ppg_acc':
        # PPG(1) + ACC(x,y,z=3) [+ optional autocorr=1]
        in_channels = 5 if dual_channel else 4

    elif use_channel in ('4ch', 'ppg_hr_accmag_maghr'):
        # [PPG, HR_line_32Hz, acc_mag, magHR_32Hz]
        in_channels = 4
        # Do not add autocorr here
        if dual_channel:
            # Ignored here (could add a warning if needed)
            pass
        # Prevent apply_autocorrelation from being enabled elsewhere
        assert not params.get('apply_autocorrelation', False), \
            "In '4ch' mode, set params['apply_autocorrelation']=False."

    else:
        raise ValueError(f"Unknown use_channel: {use_channel}")

    model = BiGRUArrhythmiaNet(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_size=hidden_size,
        dropout_p=p
    )
    return model
