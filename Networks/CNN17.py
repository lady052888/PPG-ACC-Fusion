# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0314154
# @article{bulut2025deep,
#   title={Deep CNN-based detection of cardiac rhythm disorders using PPG signals from wearable devices},
#   author={Bulut, Miray Gunay and Unal, Sencer and Hammad, Mohamed and P{\l}awiak, Pawe{\l}},
#   journal={Plos One},
#   volume={20},
#   number={2},
#   pages={e0314154},
#   year={2025},
#   publisher={Public Library of Science San Francisco, CA USA}
# }

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN17Layer(nn.Module):
    def __init__(self, input_channels=1, num_classes=3):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=50, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2),

            nn.Conv1d(128, 64, kernel_size=20, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2),

            nn.Conv1d(64, 64, kernel_size=20, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2),

            nn.Conv1d(64, 128, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2),

            nn.Conv1d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2),

            nn.Conv1d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2),
        )

        # Use dummy input to determine flattened dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 960)
            out = self.feature_extractor(dummy_input)
            self.flatten_dim = out.shape[1] * out.shape[2]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


# Factory function
def CNN17(num_classes=3):
    from config import params
    use_channel = params.get('use_channel', 'ppg_acc')
    dual_channel = params.get('dual_channel', False)

    if use_channel == 'ppg':
        in_channels = 2 if dual_channel else 1
    elif use_channel == 'acc':
        in_channels = 3
    else:  # 'ppg_acc'
        in_channels = 5 if dual_channel else 4

    return CNN17Layer(input_channels=in_channels, num_classes=num_classes)
