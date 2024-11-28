import torch
import timm
import numpy as np
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.ln = nn.LayerNorm(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x


class TimmModel(nn.Module):

    def __init__(self,
                 model_name,
                 pretrained=True,
                 img_size=383):

        super(TimmModel, self).__init__()

        self.img_size = img_size

        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        else:
            # self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self.model = timm.create_model("hf_hub:timm/convnext_base.fb_in22k_ft_in1k_384", pretrained=pretrained, num_classes=0)
            # self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, pretrained_cfg_overlay=dict(
            #     file='../weights/convnext/convnext_base_22k_1k_384.pth'))
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # --
        in_dim = 1024
        act_layer = nn.modules.activation.ReLU
        drop = 0.0
        self.back_mlp1 = Mlp(in_features=in_dim, act_layer=act_layer, drop=drop)
        self.back_mlp2 = Mlp(in_features=in_dim, act_layer=act_layer, drop=drop)

    def get_config(self, ):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def forward(self, img1, img2=None):

        if img2 is not None:
            image_features1 = self.model(img1)
            image_features2 = self.model(img2)

            vis1 = img1[18].permute(1, 2, 0).detach().cpu().numpy()
            vis2 = img2[18].permute(1, 2, 0).detach().cpu().numpy()

            # image_features1 = self.back_mlp1(self.model(img1))
            # image_features2 = self.back_mlp2(self.model(img2))

            return image_features1, image_features2

        else:
            image_features = self.model(img1)

            # image_features = self.back_mlp1(self.model(img1))

            return image_features
