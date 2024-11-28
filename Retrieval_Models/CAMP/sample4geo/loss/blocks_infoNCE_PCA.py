import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
from torch.autograd import Variable
import numpy as np
from sklearn.decomposition import PCA

def get_heartmap_pool_PCA(part_features, blocks=3, add_global=False, otherbranch=False):
    # 输入是24 144 1024
    # 需要得到的输出是24 1024 3
    heatmap = torch.mean(part_features, dim=-1)         # 把1024都压扁了 24 144
    size = part_features.size(1)          # size = 144
    batch_size = part_features.size(0)         # batch = 24

    result = []

    for i in range(batch_size):
        single_tensor = part_features[i:i+1, :, :].squeeze()
        single_tensor = single_tensor.detach().cpu().numpy()
        single_tensor = single_tensor.T

        pca = PCA(n_components=3)
        mean_single_tensor = single_tensor - np.mean(single_tensor, axis=0, keepdims=True)

        pca.fit(mean_single_tensor)
        principal_components = pca.components_
        transformed_data = pca.transform(mean_single_tensor)
        result_data = torch.from_numpy(transformed_data)
        result.append(result_data)

    part_featuers_ = torch.stack(result, dim=0)
    part_featuers_ = part_featuers_.to('cuda')

    return part_featuers_


class blocks_InfoNCE_PCA(nn.Module):

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

        # image_features1_flatten = image_features1_flatten + self.pos_embed * 2
        # image_features2_flatten = image_features2_flatten + self.pos_embed * 2

        # 需要在get_heartmap_pool里面得到 24 1024 3
        heat_result_1 = get_heartmap_pool_PCA(image_features1_flatten, blocks)      # 24 1024 3
        heat_result_2 = get_heartmap_pool_PCA(image_features2_flatten, blocks)

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
