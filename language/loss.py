import torch.nn as nn

def nll_loss(output, target):
    return nn.functional.F.nll_loss(output, target)

def cross_entrophy_loss(output, target):
    criterion = nn.CrossEntropyLoss(reduce=False)
    return criterion(output, target)
