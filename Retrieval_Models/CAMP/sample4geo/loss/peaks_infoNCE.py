import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
from torch.autograd import Variable




def get_peaks_pool(part_features, blocks=3, add_global=False, otherbranch=False):
    # 我需要让这个函数的输出保证为24 1024 3      现在的输入是24，1024，12，12

    heatmap = torch.mean(part_features, dim=1)         # 把1024都压扁了  现在这个大小是24，12，12
    shape = heatmap.shape
    res = []


    for batch_idx in range(shape[0]):

        minima_list = []
        maxima_list = []
        features_maxima_idx = []
        features_minima_idx = []
        result_feature = []

        heatmap_idx = heatmap[batch_idx, :, :]
        featuremap_idx = part_features[batch_idx, :, :, :]          # 1024, 12, 12
        for i in range(1, shape[1]-1):
            for j in range(1, shape[2]-1):
                current_value = heatmap_idx[i, j]
                neighbors_values = heatmap_idx[i-1:i+2, j-1:j+2].flatten()

                # 判断是否为极小值点
                if current_value <= neighbors_values.min():
                    minima_list.append((i, j))

                # 判断是否为极大值点
                if current_value >= neighbors_values.max():
                    maxima_list.append((i, j))

        # i=0时，跟下面的比：
        for j in range(1, shape[2]-1):
            current_value = heatmap_idx[0, j]
            neighbors_values = heatmap_idx[0:2, j-1:j+2].flatten()

            # 判断是否为极小值点
            if current_value <= neighbors_values.min():
                minima_list.append((0, j))

            # 判断是否为极大值点
            if current_value >= neighbors_values.max():
                maxima_list.append((0, j))

        # i=最大值时，跟上面的比：
        for j in range(1, shape[2]-1):
            current_value = heatmap_idx[shape[1]-1, j]
            neighbors_values = heatmap_idx[shape[1]-2:shape[1], j-1:j+2].flatten()

            # 判断是否为极小值点
            if current_value <= neighbors_values.min():
                minima_list.append((shape[1]-1, j))

            # 判断是否为极大值点
            if current_value >= neighbors_values.max():
                maxima_list.append((shape[1]-1, j))

        # j=0时，跟右边的比
        for i in range(1, shape[1]-1):
            current_value = heatmap_idx[i, 0]
            neighbors_values = heatmap_idx[i-1:i+2, 0:2].flatten()

            # 判断是否为极小值点
            if current_value <= neighbors_values.min():
                minima_list.append((i, 0))

            # 判断是否为极大值点
            if current_value >= neighbors_values.max():
                maxima_list.append((i, 0))

        # j=最大值，跟左边的比
        for i in range(1, shape[1]-1):
            current_value = heatmap_idx[i, shape[2]-1]
            neighbors_values = heatmap_idx[i-1:i+2, shape[2]-2:shape[2]].flatten()

            # 判断是否为极小值点
            if current_value <= neighbors_values.min():
                minima_list.append((i, shape[2]-1))

            # 判断是否为极大值点
            if current_value >= neighbors_values.max():
                maxima_list.append((i, shape[2]-1))

        for ma in range(len(maxima_list)):
            features_maxima_idx.append(featuremap_idx[:, maxima_list[ma][0], maxima_list[ma][1]])

        for mi in range(len(minima_list)):
            features_minima_idx.append(featuremap_idx[:, minima_list[mi][0], minima_list[mi][1]])

        features_maxima_idx = torch.stack(features_maxima_idx)  # 5*1024
        features_minima_idx = torch.stack(features_minima_idx)  # 7*1024

        features_maxima_idx = torch.mean(features_maxima_idx, dim=0)
        features_minima_idx = torch.mean(features_minima_idx, dim=0)
        features_gap_idx = featuremap_idx.view(featuremap_idx.shape[0], -1)
        features_gap_idx = torch.mean(features_gap_idx, dim=1)

        result_feature.append(features_maxima_idx)
        result_feature.append(features_minima_idx)
        result_feature.append(features_gap_idx)
        result_feature = torch.stack(result_feature)

        res.append(result_feature)

    res = torch.stack(res)
    res = torch.transpose(res, 1, 2)
    return res

    # # for i in range(shape[1]):
    # #     for j in range(shape[2]):
    #
    # size = part_features.size(1)
    # arg = torch.argsort(heatmap, dim=1, descending=True)
    # x_sort = [part_features[i, arg[i], :] for i in range(part_features.size(0))]
    # x_sort = torch.stack(x_sort, dim=0)
    #
    # # -- 按照地物自动聚类的类别数来将16*16的区域进行分类
    # split_each = size / blocks
    # split_list = [int(split_each) for i in range(blocks - 1)]
    # split_list.append(size - sum(split_list))
    # split_x = x_sort.split(split_list, dim=1)
    #
    # split_list = [torch.mean(split, dim=1) for split in split_x]
    # part_featuers_ = torch.stack(split_list, dim=2)
    # if add_global:
    #     global_feat = torch.mean(part_features, dim=1).view(part_features.size(0), -1, 1).expand(-1, -1, blocks)
    #     part_featuers_ = part_featuers_ + global_feat
    # if otherbranch:
    #     otherbranch_ = torch.mean(torch.stack(split_list[1:], dim=2), dim=-1)
    #     return part_featuers_, otherbranch_
    # return part_featuers_


class peaks_InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.loss_function = loss_function  # -- default CrossEntropy
        self.device = device

    # image_features1大小为24，1024，12，12
    def forward(self, image_features1, image_features2, logit_scale):
        peak_result_1 = get_peaks_pool(image_features1)      # 我需要让这个函数的输出保证为24 1024 3
        peak_result_2 = get_peaks_pool(image_features2)      # 后面的代码不用改

        # 1. concate
        if 1:
            image_features_peaks_1 = torch.cat((peak_result_1[:, :, 0], peak_result_1[:, :, 1], peak_result_1[:, :, 2]), dim=-1)
            image_features_peaks_2 = torch.cat((peak_result_2[:, :, 0], peak_result_2[:, :, 1], peak_result_2[:, :, 2]), dim=-1)

            image_features1 = F.normalize(image_features_peaks_1, dim=-1)
            image_features2 = F.normalize(image_features_peaks_2, dim=-1)

            logits_per_image1 = logit_scale * image_features1 @ image_features2.T

            logits_per_image2 = logits_per_image1.T

            labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)

            loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels)) / 2

        # 2. weight and sum (效果更差)
        # image_features_blocks_1 = (3 * weights[0] * heat_result_1[:, :, 0] +
        #                            2 * weights[1] * heat_result_1[:, :, 1] + 1 * weights[2] * heat_result_1[:, :, 2])
        # image_features_blocks_2 = (3 * weights[0] * heat_result_2[:, :, 0] +
        #                            2 * weights[1] * heat_result_2[:, :, 1] + 1 * weights[2] * heat_result_2[:, :, 2])

        # 3. multi-head loss
        else:
            image_features_blocks_1_1, image_features_blocks_1_2, image_features_blocks_1_3 =\
                heat_result_1[:, :, 0], heat_result_1[:, :, 1], heat_result_1[:, :, 2]
            image_features_blocks_2_1, image_features_blocks_2_2, image_features_blocks_2_3 = \
                heat_result_2[:, :, 0], heat_result_2[:, :, 1], heat_result_2[:, :, 2]

            image_features1_1, image_features1_2, image_features1_3 =\
                F.normalize(image_features_blocks_1_1, dim=-1), F.normalize(image_features_blocks_1_2, dim=-1), F.normalize(image_features_blocks_1_3, dim=-1)
            image_features2_1, image_features2_2, image_features2_3 = \
                F.normalize(image_features_blocks_2_1, dim=-1), F.normalize(image_features_blocks_2_2, dim=-1), F.normalize(image_features_blocks_2_3, dim=-1)

            #--
            logits_per_image1_1 = logit_scale * image_features1_1 @ image_features2_1.T
            logits_per_image2_1 = logits_per_image1_1.T
            labels = torch.arange(len(logits_per_image1_1), dtype=torch.long, device=self.device)
            loss1 = (self.loss_function(logits_per_image1_1, labels) + self.loss_function(logits_per_image2_1,
                                                                                          labels)) / 2

            logits_per_image1_2 = logit_scale * image_features1_2 @ image_features2_2.T
            logits_per_image2_2 = logits_per_image1_2.T
            labels = torch.arange(len(logits_per_image1_2), dtype=torch.long, device=self.device)
            loss2 = (self.loss_function(logits_per_image1_2, labels) + self.loss_function(logits_per_image2_2,
                                                                                          labels)) / 2

            logits_per_image1_3 = logit_scale * image_features1_3 @ image_features2_3.T
            logits_per_image2_3 = logits_per_image1_3.T
            labels = torch.arange(len(logits_per_image1_3), dtype=torch.long, device=self.device)
            loss3 = (self.loss_function(logits_per_image1_3, labels) + self.loss_function(logits_per_image2_3,
                                                                                          labels)) / 2

            loss = (loss1 + loss2 + loss3) / 3

        return loss
