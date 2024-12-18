import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class DiceFocalLoss(nn.Module):
    def __init__(self):
        super(DiceFocalLoss, self).__init__()
        self.focalloss = FocalLoss()

    def forward(self, output, y_true, compare_rule='>', thr=0.95):
        smooth = 1e-9
        num = output.size(1)
        loss = 0
        dice_loss = 0
        for c in range(0, num):
            loss += self.focalloss(output[:, c], y_true[:, c])
            pred_flat = output[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc = (2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)  # dsc系数
            dice_loss += 1 - dsc
        return (loss + dice_loss) / num

class OnlyFocalLoss(nn.Module):
    def __init__(self):
        super(OnlyFocalLoss, self).__init__()
        self.focalloss = FocalLoss()

    def forward(self, output, y_true):
        num = output.size(1)
        loss = 0
        for c in range(0, num):
            loss += self.focalloss(output[:, c], y_true[:, c])
        return loss / num


class ComboLoss(nn.Module):
    def __init__(self):
        super(ComboLoss, self).__init__()
        self.focalloss = FocalLoss()

    def forward(self, output, y_true):
        smooth = 1e-9
        num = output.size(1)
        loss_fl = 0
        loss_ce = 0
        dice_loss = 0
        for c in range(0, num):
            loss_fl += self.focalloss(output[:, c], y_true[:, c])
            loss_ce += F.binary_cross_entropy(output[:, c], y_true[:, c])
            pred_flat = output[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc = (2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)  # dsc系数
            dice_loss += 1 - dsc
        return (loss_ce + loss_fl + dice_loss) / num


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict)  # sigmoide获取概率
        # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (
                1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
