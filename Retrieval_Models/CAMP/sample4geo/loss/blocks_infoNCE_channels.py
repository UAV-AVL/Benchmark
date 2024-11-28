import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
from torch.autograd import Variable

def get_heartmap_pool_channels(part_features, blocks=3, add_global=False, otherbranch=False):
    # 需要在get_heartmap_pool_channels里得到24，144，3
    # 输入的张量大小是24 144 1024
    heatmap = torch.mean(part_features, dim=-2)         # 把144都压扁了
    size = part_features.size(2)                        # size是1024
    arg = torch.argsort(heatmap, dim=1, descending=True)    # 对1024排序
    x_sort = [part_features[i, :, arg[i]] for i in range(part_features.size(0))]    # 排序之后的features
    x_sort = torch.stack(x_sort, dim=0)

    # -- 按照地物自动聚类的类别数来将1024维度通道进行分类

    split_each = size / blocks
    split_list = [int(split_each) for i in range(blocks - 1)]
    split_list.append(size - sum(split_list))
    split_x = x_sort.split(split_list, dim=2)

    split_list = [torch.mean(split, dim=2) for split in split_x]
    part_featuers_ = torch.stack(split_list, dim=1)     # 不确定这里是否dim=2

    return part_featuers_


class blocks_InfoNCE_channels(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.loss_function = loss_function  # -- default CrossEntropy
        self.device = device
        # self.pos_embed = nn.Parameter(torch.zeros(1, 144, 1024))
        # self.pos_embed = torch.tensor(self.pos_embed)
        # trunc_normal_(self.pos_embed, std=.02)

    def forward(self, image_features1, image_features2, logit_scale, weights, blocks=3):

        image_features1_flatten = image_features1.view(image_features1.size(0), image_features1.size(1), -1).transpose(
            -2, -1)             # 24 144 1024
        image_features2_flatten = image_features1.view(image_features2.size(0), image_features2.size(1), -1).transpose(
            -2, -1)             # 24 144 1024
        # 到这里还没问题，需要在get_heartmap_pool_channels里得到24，144，3

        heat_result_1 = get_heartmap_pool_channels(image_features1_flatten, blocks)      # 24, 3, 144
        heat_result_2 = get_heartmap_pool_channels(image_features2_flatten, blocks)
        heat_result_1 = torch.transpose(heat_result_1, 1, 2)                             # 24, 144, 3
        heat_result_2 = torch.transpose(heat_result_2, 1, 2)

        # 1. concate
        if 1:
            image_features_blocks_1 = torch.cat((heat_result_1[:, :, 0], heat_result_1[:, :, 1], heat_result_1[:, :, 2]),
                                                dim=-1)
            image_features_blocks_2 = torch.cat((heat_result_2[:, :, 0], heat_result_2[:, :, 1], heat_result_2[:, :, 2]),
                                                dim=-1)

            # image_features_blocks_1 = torch.cat((heat_result_1[:, :, 0], heat_result_1[:, :, 1]), dim=-1)
            # image_features_blocks_2 = torch.cat((heat_result_2[:, :, 0], heat_result_2[:, :, 1]), dim=-1)

            image_features1 = F.normalize(image_features_blocks_1, dim=-1)
            image_features2 = F.normalize(image_features_blocks_2, dim=-1)

            logits_per_image1 = logit_scale * image_features1 @ image_features2.T

            logits_per_image2 = logits_per_image1.T

            labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)

            loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels)) / 2


        return loss
