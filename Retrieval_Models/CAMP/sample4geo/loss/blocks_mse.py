import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
from torch.autograd import Variable

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


class blocks_mse(nn.Module):

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.device = device


    def forward(self, image_features1, image_features2, logit_scale, weights, blocks=3):

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

            loss = torch.nn.functional.mse_loss(image_features1, image_features2)

        return loss
