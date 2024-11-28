import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
from torch.autograd import Variable

import torch.nn.functional as F
from torch.autograd import Variable
import torch


def cal_loss(outputs, labels, loss_func):
    loss = 0
    if isinstance(outputs, list):
        for i in outputs:
            loss += loss_func(i, labels)
        loss = loss / len(outputs)
    else:
        loss = loss_func(outputs, labels)
    return loss


def cal_kl_loss(outputs, outputs2, loss_func):
    loss = 0
    if isinstance(outputs, list):
        for i in range(len(outputs)):
            loss += loss_func(F.log_softmax(outputs[i], dim=1),
                              F.softmax(Variable(outputs2[i]), dim=1))
        loss = loss / len(outputs)
    else:
        loss = loss_func(F.log_softmax(outputs, dim=1),
                         F.softmax(Variable(outputs2), dim=1))
    return loss


def cal_triplet_loss(outputs, outputs2, labels, loss_func, split_num=8):
    if isinstance(outputs, list):
        loss = 0
        for i in range(len(outputs)):
            out_concat = torch.cat((outputs[i], outputs2[i]), dim=0)
            labels_concat = torch.cat((labels, labels), dim=0)
            loss += loss_func(out_concat, labels_concat)
        loss = loss / len(outputs)
    else:
        out_concat = torch.cat((outputs, outputs2), dim=0)
        labels_concat = torch.cat((labels, labels), dim=0)
        loss = loss_func(out_concat, labels_concat)
    return loss

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-6)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-6).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        # dist_mat = cosine_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss


class Tripletloss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, hard_factor=0.0):
        super(Tripletloss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.hard_factor = hard_factor

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """

        n = inputs.size(0)

        inputs = normalize(inputs,axis=-1)

        dist =euclidean_dist(inputs,inputs)
        # dist =cosine_dist(inputs,inputs)
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            if i < n/2:
                dist_ap.append(dist[i][int(n/2):n][mask[i][int(n/2):n]].max().unsqueeze(0))
                dist_an.append(dist[i][int(n/2):n][(mask[i] == 0)[int(n/2):n]].min().unsqueeze(0))
            else:
                dist_ap.append(dist[i][0:int(n/2)][mask[i][0:int(n/2)]].max().unsqueeze(0))
                dist_an.append(dist[i][0:int(n/2)][(mask[i] == 0)[0:int(n/2)]].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)

        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def get_heartmap_pool(part_features, blocks=3, add_global=False, otherbranch=False):
    # 输入是24 144 1024
    # 需要得到的输出是24 1024 3
    heatmap = torch.mean(part_features, dim=-1)         # 把1024都压扁了
    size = part_features.size(1)    # size=144
    arg = torch.argsort(heatmap, dim=1, descending=True)
    x_sort = [part_features[i, arg[i], :] for i in range(part_features.size(0))]
    x_sort = torch.stack(x_sort, dim=0)

    # -- 按照地物自动聚类的类别数来将16*16的区域进行分类

    split_each = size / blocks
    split_list = [int(split_each) for i in range(blocks - 1)]
    split_list.append(size - sum(split_list))
    split_x = x_sort.split(split_list, dim=1)

    split_list = [torch.mean(split, dim=1) for split in split_x]
    part_featuers_ = torch.stack(split_list, dim=2)

    return part_featuers_


class triplet_loss(nn.Module):

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        # self.pos_embed = nn.Parameter(torch.zeros(1, 144, 1024))
        # self.pos_embed = torch.tensor(self.pos_embed)
        # trunc_normal_(self.pos_embed, std=.02)

    def forward(self, image_features1, image_features2, labels, blocks=3):

        image_features1_flatten = image_features1.view(image_features1.size(0), image_features1.size(1), -1).transpose(
            -2, -1)             # 24 144 1024
        image_features2_flatten = image_features2.view(image_features2.size(0), image_features2.size(1), -1).transpose(
            -2, -1)             # 24 144 1024

        # image_features1_flatten = image_features1_flatten + self.pos_embed * 2
        # image_features2_flatten = image_features2_flatten + self.pos_embed * 2

        # 需要在get_heartmap_pool里面得到 24 1024 3
        heat_result_1 = get_heartmap_pool(image_features1_flatten, blocks)      # 24 1024 3
        heat_result_2 = get_heartmap_pool(image_features2_flatten, blocks)

        # 1. concate
        if 1:
            channels1 = [heat_result_1[:, :, i] for i in range(heat_result_1.shape[2])]
            channels2 = [heat_result_2[:, :, i] for i in range(heat_result_2.shape[2])]

            # 使用 torch.cat 连接所有通道
            image_features_blocks_1 = torch.cat(channels1, dim=-1)
            image_features_blocks_2 = torch.cat(channels2, dim=-1)

            # image_features_blocks_1 = torch.cat((heat_result_1[:, :, 0], heat_result_1[:, :, 1], heat_result_1[:, :, 2]),
            #                                     dim=-1)
            # image_features_blocks_2 = torch.cat((heat_result_2[:, :, 0], heat_result_2[:, :, 1], heat_result_2[:, :, 2]),
            #                                     dim=-1)

            # image_features_blocks_1 = torch.cat((heat_result_1[:, :, 0], heat_result_1[:, :, 1]), dim=-1)
            # image_features_blocks_2 = torch.cat((heat_result_2[:, :, 0], heat_result_2[:, :, 1]), dim=-1)

            image_features1 = F.normalize(image_features_blocks_1, dim=-1)
            image_features2 = F.normalize(image_features_blocks_2, dim=-1)

            loss_fn2 = Tripletloss(margin=0.3)

            loss = cal_triplet_loss(image_features1, image_features2, labels, loss_fn2)

            #
            # logits_per_image1 = logit_scale * image_features1 @ image_features2.T
            #
            # logits_per_image2 = logits_per_image1.T
            #
            # labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
            #
            # loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels)) / 2


        return loss

