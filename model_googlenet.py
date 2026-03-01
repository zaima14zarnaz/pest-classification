import torch
import torch.nn as nn
from torchvision.models import googlenet


def build_googlenet_ip102(num_classes: int, aux_logits: bool = True):
    model = googlenet(weights=None, aux_logits=aux_logits)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if aux_logits:
        # torchvision GoogLeNet defines these when aux_logits=True
        model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_classes)
        model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes)

    return model
