import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# proposed models: Zhao25-RhythmiNet -> ResNet10_TemporalAttention_SE; proposed model -> ResNet10_TemporalAttention_DilatedL2
from Networks.ResNet10_TemporalAttention_SE import ResNet10_TemporalAttention_SE
from Networks.ResNet10_TemporalAttention_DilatedL2 import ResNet10_TemporalAttention_DilatedL2

# reproduced models
from Networks.resnet import resnet1D18, resnet1D34,resnet1D50,resnet1D101
from Networks.MobileNet import mobilenet_1d

from Networks.VGG16 import VGG16_1D_Multimodal
from Networks.BiGRU2025 import BiGRU_Multimodal
from Networks.KDD2019 import KDD2019
from Networks.CNN17 import CNN17

import math
from config import params, get_num_classes 


def _get_class_weights():
    weights = params.get('class_weights')
    if weights is None:
        return None
    device = params.get('device', 'cpu')
    return torch.tensor(weights, dtype=torch.float, device=device)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = log_probs.exp()
        targets = targets.long()

        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        if self.weight is not None:
            w = self.weight.gather(0, targets)
        else:
            w = 1.0

        loss = -w * ((1.0 - p_t) ** self.gamma) * log_p_t

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

def get_model(model_name, device):
    num_classes = get_num_classes()
    
    # reproduce (paper naming)
    if model_name == 'Shen19-50CNN':
        model = KDD2019(num_classes=num_classes).to(device)
    elif model_name == 'Bulut25-CNN17':
        model = CNN17(num_classes=num_classes).to(device)
    elif model_name == 'Han25-BiGRU':
        model = BiGRU_Multimodal(num_classes=num_classes).to(device)
    elif model_name == 'Liu22-DCNN':
        model = VGG16_1D_Multimodal(num_classes=num_classes).to(device)

    # my model 2025
    elif model_name == 'Zhao25-RhythmiNet':
        model = ResNet10_TemporalAttention_SE(num_classes=num_classes).to(device)
    elif model_name == 'ResNet10_TemporalAttention_DilatedL2':
        model = ResNet10_TemporalAttention_DilatedL2(num_classes=num_classes).to(device)

   # some classic models
    elif model_name == 'resnet18':
        model = resnet1D18(num_classes=num_classes).to(device)
    elif model_name == 'resnet34':
        model = resnet1D34(num_classes=num_classes).to(device)
    elif model_name == 'resnet50':
        model = resnet1D50(num_classes=num_classes).to(device)
    elif model_name == 'resnet101':
        model = resnet1D101(num_classes=num_classes).to(device)
    elif model_name == 'mobile_net':
        model = mobilenet_1d(num_classes=num_classes).to(device)
    

    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    print(model)
    return model


def get_optimizer(optimizer_name, model, lr=0.001, weight_decay=0, momentum=0.9):
    'SGD', 'Adam','AdamW'
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer name: {optimizer_name}")
    
    return optimizer


def get_scheduler(scheduler_name, optimizer, step_size=10, gamma=0.1, patience=5):
    
    if scheduler_name == 'StepLR':
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience)
    elif scheduler_name == 'ExponentialLR':    
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    elif scheduler_name == 'warmup_scheduler':  
       # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: ((1 + math.cos(x * math.pi / params['num_epochs'])) / 2) * (1 - params['lrf']) + params['lrf']  # cosine
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    else:
        raise ValueError(f"Unknown scheduler name: {scheduler_name}")
    
    return scheduler


def get_criterion(criterion_name):
    
    if criterion_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif criterion_name == 'WeightedCrossEntropyLoss':
        weights = _get_class_weights()
        if weights is None:
            raise ValueError("class_weights not set in config params for WeightedCrossEntropyLoss")
        return nn.CrossEntropyLoss(weight=weights)
    # elif criterion_name == 'MSELoss':
    #     return nn.MSELoss()
    elif criterion_name == 'NLLLoss':
        return nn.NLLLoss()
    elif criterion_name == 'BCELoss':
        return nn.BCELoss()
    elif criterion_name == 'FocalLoss':
        weights = _get_class_weights()
        gamma = params.get('focal_gamma', 2.0)
        reduction = params.get('focal_reduction', 'mean')
        return FocalLoss(gamma=gamma, weight=weights, reduction=reduction)

    else:
        raise ValueError(f"Unknown criterion name: {criterion_name}")
