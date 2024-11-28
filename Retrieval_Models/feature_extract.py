import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.cuda.amp import autocast
import torch.nn.functional as F

def extract_features(retrieval_size, config, method_dict, MID, block_size, SATELLITE, UAV, val_transforms, model):
    model.eval()

    row, column = MID.shape
    process = list(range(row))
    img_features_list = []
    mids = MID.reshape(-1,2)
    # Process images in batches
    for batch_start in tqdm(range(0, len(process), config['BATCH_SIZE']), desc='Image Retrieval'):

        # Get the end index of the current batch
        batch_end = batch_start + config['BATCH_SIZE']
        # If the batch end index exceeds the total number, use the actual number
        if batch_end > len(process):
            batch_end = len(process)

        batch_indices = process[batch_start:batch_end]
        batch_images = []

        for item in batch_indices:
            mid_x, mid_y = mids[item, :]
            img = cv2.resize(SATELLITE[int(mid_x - block_size[0] / 2):int(mid_x + block_size[0] / 2),
                  int(mid_y - block_size[1] / 2):int(mid_y + block_size[1] / 2)], (retrieval_size,retrieval_size))
            img = Image.fromarray(img)
            img = val_transforms(img)
            batch_images.append(img)

        # Convert images to tensors and stack them
        if len(batch_images) > 0:  # Ensure there is at least one image
            batch_images = torch.stack(batch_images, dim=0).to(config['DEVICE'])
            with torch.no_grad():
                with torch.cuda.amp.autocast():

                    batch_features = model(batch_images)

                    batch_features = feature_fusion_all(method_dict, batch_features)

                    if config['RETRIEVAL_FEATURE_NORM']:
                        batch_features = F.normalize(batch_features, dim=-1)

            # Append features to the list
            img_features_list.append(batch_features.to(torch.float32))

    # Concatenate all features together
    if len(img_features_list) > 0:  # Ensure there is at least one feature
        img_features = torch.cat(img_features_list, dim=0)
    else:
        img_features = torch.tensor([], dtype=torch.float32, device=config['DEVICE'])

    with torch.no_grad():
        with autocast():
            UAV1 = UAV.to(config['DEVICE'])
            UAV1 = UAV1.unsqueeze(0)

            UAV_feature = model(UAV1)
            UAV_feature = feature_fusion_all(method_dict, UAV_feature)
            if config['RETRIEVAL_FEATURE_NORM']:
                UAV_feature = F.normalize(UAV_feature, dim=-1)

    return img_features, UAV_feature

def feature_fusion_all(method_dict, img_feature):
    if method_dict['retrieval_method'] == 'MCCG':
        if len(img_feature.shape) == 3:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True) * np.sqrt(img_feature.size(-1))
            img_feature = img_feature.div(fnorm.expand_as(img_feature))
            img_feature = img_feature.view(img_feature.size(0), -1)
        else:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True)
            img_feature = img_feature.div(fnorm.expand_as(img_feature))
    elif method_dict['retrieval_method'] == 'LPN':
        block = 6
        fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True) * np.sqrt(block)
        img_feature = img_feature.div(fnorm.expand_as(img_feature))
        img_feature = img_feature.view(img_feature.size(0), -1)
    elif method_dict['retrieval_method'] == 'MIFT':
        if len(img_feature.shape)==3:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True) * np.sqrt(img_feature.size(-1))
            img_feature = img_feature.div(fnorm.expand_as(img_feature))
            img_feature = img_feature.view(img_feature.size(0), -1)
        else:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True)
            img_feature = img_feature.div(fnorm.expand_as(img_feature))
    elif method_dict['retrieval_method'] == 'FSRA':
        if len(img_feature.shape)==3:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True) * np.sqrt(img_feature.size(-1))
            img_feature = img_feature.div(fnorm.expand_as(img_feature))
            img_feature = img_feature.view(img_feature.size(0), -1)
        else:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True)
            img_feature = img_feature.div(fnorm.expand_as(img_feature))

    elif method_dict['retrieval_method'] == 'CAMP':
            img_feature = img_feature[-2]
    elif method_dict['retrieval_method'] == 'DAC':
            img_feature = img_feature[-2]
    return img_feature