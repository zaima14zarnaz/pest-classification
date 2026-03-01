import torch.nn as nn
from torchvision.models import vgg11, vgg13, vgg16, vgg19
from torchvision.models import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

_VGG_VARIANTS = {
    "vgg11": vgg11,
    "vgg13": vgg13,
    "vgg16": vgg16,
    "vgg19": vgg19,
    "vgg11_bn": vgg11_bn,
    "vgg13_bn": vgg13_bn,
    "vgg16_bn": vgg16_bn,
    "vgg19_bn": vgg19_bn,
}

def build_vggnet_ip102(num_classes: int, variant: str = "vgg16_bn", pretrained: bool = False):
    if variant not in _VGG_VARIANTS:
        raise ValueError(f"Unknown VGG variant: {variant}. Options: {list(_VGG_VARIANTS.keys())}")

    # torchvision vgg uses weights=... in newer versions; weights=None is fine for from-scratch
    # If you want pretrained ImageNet weights, you can switch this to the proper Weights enum.
    model_fn = _VGG_VARIANTS[variant]
    model = model_fn(weights=None if not pretrained else None)

    # Replace final classifier layer
    # VGG: classifier = [Linear(25088->4096), ReLU, Dropout, Linear(4096->4096), ReLU, Dropout, Linear(4096->1000)]
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model