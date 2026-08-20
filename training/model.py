"""
Model definition for CheckpointLab.

Uses ResNet-18 adapted for CIFAR-10 (10 classes, 32x32 images). We use the
torchvision ResNet-18 architecture but swap the first conv/maxpool so it
doesn't immediately downsample tiny 32x32 images to nothing.
"""
import torch.nn as nn
from torchvision.models import resnet18


def build_model(num_classes: int = 10) -> nn.Module:
    model = resnet18(weights=None, num_classes=num_classes)

    # Standard ResNet-18 was designed for 224x224 ImageNet images. CIFAR-10
    # images are 32x32, so the default 7x7 stride-2 conv + maxpool destroys
    # almost all spatial information before the first residual block even
    # runs. Replace with a 3x3 stride-1 conv and drop the maxpool, which is
    # the standard adaptation used for CIFAR-scale ResNets.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    return model
