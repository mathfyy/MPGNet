import torch
import torch.nn as nn


class TVLoss_2d_edge(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss_2d_edge, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x, L):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])

        h_L = torch.pow((L[:, :, 1:, :] - L[:, :, :h_x - 1, :]), 2)
        w_L = torch.pow((L[:, :, :, 1:] - L[:, :, :, :w_x - 1]), 2)

        h_t = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2)
        w_t = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2)

        h_tv = torch.mul(h_t, (1 - h_L)).sum()
        w_tv = torch.mul(w_t, (1 - w_L)).sum()

        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class TVLoss_2d(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss_2d, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])

        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class TVLoss_3d(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss_3d, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        d_x = x.size()[2]
        h_x = x.size()[3]
        w_x = x.size()[4]
        count_d = self._tensor_size(x[:, :, 1:, :, :])
        count_h = self._tensor_size(x[:, :, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, :, 1:])

        d_tv = torch.pow((x[:, :, 1:, :, :] - x[:, :, :d_x - 1, :, :]), 2).sum()
        h_tv = torch.pow((x[:, :, :, 1:, :] - x[:, :, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, :, 1:] - x[:, :, :, :, :w_x - 1]), 2).sum()

        return self.TVLoss_weight * 2 * (d_tv / count_d + h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3] * t.size()[4]


class TVLoss_3d_edge(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss_3d_edge, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x, L):
        batch_size = x.size()[0]
        d_x = x.size()[2]
        h_x = x.size()[3]
        w_x = x.size()[4]
        count_d = self._tensor_size(x[:, :, 1:, :, :])
        count_h = self._tensor_size(x[:, :, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, :, 1:])

        d_L = torch.pow((L[:, :, 1:, :, :] - L[:, :, :d_x - 1, :, :]), 2)
        h_L = torch.pow((L[:, :, :, 1:, :] - L[:, :, :, :h_x - 1, :]), 2)
        w_L = torch.pow((L[:, :, :, :, 1:] - L[:, :, :, :, :w_x - 1]), 2)

        d_t = torch.pow((x[:, :, 1:, :, :] - x[:, :, :d_x - 1, :, :]), 2)
        h_t = torch.pow((x[:, :, :, 1:, :] - x[:, :, :, :h_x - 1, :]), 2)
        w_t = torch.pow((x[:, :, :, :, 1:] - x[:, :, :, :, :w_x - 1]), 2)

        d_tv = torch.mul(d_t, (1 - d_L)).sum()
        h_tv = torch.mul(h_t, (1 - h_L)).sum()
        w_tv = torch.mul(w_t, (1 - w_L)).sum()

        return self.TVLoss_weight * 2 * (d_tv / count_d + h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3] * t.size()[4]
