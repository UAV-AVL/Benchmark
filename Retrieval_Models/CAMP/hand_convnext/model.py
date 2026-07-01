import torch.nn as nn
from .ConvNext import make_convnext_model
import torch
import numpy as np

class two_view_net(nn.Module):
    def __init__(self, class_num, block=4, return_f=False, resnet=False, pos_scale=0.1, if_learn_ECE_weight=False,
                             learn_weight_D_D=None, learn_weight_S_S=None,
                             learn_weight_D_fine_D_fine=None,
                             learn_weight_D_fine_S_fine=None,
                             learn_weight_S_fine_S_fine=None):
        super(two_view_net, self).__init__()
        self.model_1 = make_convnext_model(num_class=class_num, block=block, return_f=return_f, resnet=resnet, pos_scale=pos_scale)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))# 0.07
        if if_learn_ECE_weight:
            self.ECE_weight_D_D = torch.nn.Parameter(torch.ones([]) * learn_weight_D_D)
            self.ECE_weight_S_S = torch.nn.Parameter(torch.ones([]) * learn_weight_S_S)
            self.ECE_weight_D_fine_D_fine = torch.nn.Parameter(torch.ones([]) * learn_weight_D_fine_D_fine)
            self.ECE_weight_S_fine_S_fine = torch.nn.Parameter(torch.ones([]) * learn_weight_S_fine_S_fine)
            self.ECE_weight_D_fine_S_fine = torch.nn.Parameter(torch.ones([]) * learn_weight_D_fine_S_fine)

        else:
            self.ECE_weight_D_D = torch.nn.Parameter(torch.ones([]) * 1.0)
            self.ECE_weight_S_S = torch.nn.Parameter(torch.ones([]) * 1.0)
            self.ECE_weight_D_fine_D_fine = torch.nn.Parameter(torch.ones([]) * 1.0)
            self.ECE_weight_D_fine_S_fine = torch.nn.Parameter(torch.ones([]) * 1.0)
            self.ECE_weight_S_fine_S_fine = torch.nn.Parameter(torch.ones([]) * 1.0)


        self.logit_scale_blocks = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # 2. weight for blocks_infoNCE
        self.w_blocks1 = torch.nn.Parameter(torch.ones([]))
        self.w_blocks2 = torch.nn.Parameter(torch.ones([]))
        self.w_blocks3 = torch.nn.Parameter(torch.ones([]))

    def get_config(self):
        input_size = (3, 224, 224)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        config = {
            'input_size': input_size,
            'mean': mean,
            'std': std
        }
        return config

    def forward(self, x1, x2=None):

        if x2 is not None:
            y1 = self.model_1(x1)
            y2 = self.model_1(x2)
            return y1, y2
        else:
            y1 = self.model_1(x1)
            return y1


def make_model(opt):
    if opt.views == 2:
        model = two_view_net(opt.nclasses, block=opt.block, return_f=opt.triplet_loss, resnet=opt.resnet,
                             pos_scale=opt.pos_scale, if_learn_ECE_weight=opt.if_learn_ECE_weights,
                             learn_weight_D_D=opt.learn_weight_D_D, learn_weight_S_S=opt.learn_weight_S_S,
                             learn_weight_D_fine_D_fine=opt.learn_weight_D_fine_D_fine,
                             learn_weight_D_fine_S_fine=opt.learn_weight_D_fine_S_fine,
                             learn_weight_S_fine_S_fine=opt.learn_weight_S_fine_S_fine)

    return model
