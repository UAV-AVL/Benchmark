import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.nn as nn


def train(train_config, model, dataloader, loss_functions, optimizer, epoch, train_steps_per, tensorboard=None,
          scheduler=None, scaler=None):
    # set model train mode
    model.train()

    losses = AverageMeter()

    # wait before starting progress bar
    time.sleep(0.1)

    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)

    step = 1

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    criterion = nn.CrossEntropyLoss()
    wq_logit = train_config.infoNCE_logit
    wq_logit = torch.tensor(wq_logit)

    # for loop over one epoch
    for query, reference, ids, labels in bar:

        if scaler:
            with (autocast()):  # -- 使用混合精度
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                labels = labels.to(train_config.device)

                # Forward pass
                if train_config.handcraft_model is not True:
                    features1, features2 = model(query, reference)
                else:
                    output1, output2 = model(query, reference)
                    features1, features2 = output1[-2], output2[-2]  # -- for contrastive
                    features_fine_1, features_fine_2 = output1[-1], output2[-1]  # -- for fine-grained

                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                    loss = loss_functions["infoNCE"](features1, features2, model.module.logit_scale.exp())
                else:
                    # 1. infoNCE
                    loss = loss_functions["infoNCE"](features1, features2, model.logit_scale.exp())
                    loss_D_D = loss_functions["infoNCE"](features1, features1, model.logit_scale.exp())
                    loss_S_S = loss_functions["infoNCE"](features2, features2, model.logit_scale.exp())


                    # 2. Fine-grained
                    blocks = train_config.blocks_for_PPB
                    weights = [model.w_blocks1, model.w_blocks2, model.w_blocks3]

                    # ========================================================================================
                    loss_D_fine_S_fine = loss_functions["blocks_mse"](features_fine_1, features_fine_2,
                                                                      model.logit_scale_blocks.exp(), weights,
                                                                      blocks)
                    # ========================================================================================


                    loss_D_fine_D_fine = loss_functions["blocks_infoNCE"](features_fine_1, features_fine_1,
                                                                          model.logit_scale_blocks.exp(), weights,
                                                                          blocks)
                    loss_S_fine_S_fine = loss_functions["blocks_infoNCE"](features_fine_2, features_fine_2,
                                                                          model.logit_scale_blocks.exp(), weights,
                                                                          blocks)


                if train_config.if_learn_ECE_weights:

                    if train_config.if_use_plus_1:
                        if train_config.only_DS:
                            lossall = train_config.weight_D_S * loss + \
                                      model.ECE_weight_D_D * loss_D_D + \
                                      (1 - model.ECE_weight_D_D) * loss_S_S + \
                                      train_config.weight_D_fine_S_fine * loss_D_fine_S_fine + \
                                      0. * loss_D_fine_D_fine + \
                                      0. * loss_S_fine_S_fine

                        elif train_config.only_fine:
                            lossall = train_config.weight_D_S * loss + \
                                      0. * loss_D_D + \
                                      0. * loss_S_S + \
                                      train_config.weight_D_fine_S_fine * loss_D_fine_S_fine + \
                                      model.ECE_weight_D_fine_D_fine * loss_D_fine_D_fine + \
                                      (1 - model.ECE_weight_D_fine_D_fine) * loss_S_fine_S_fine

                        elif train_config.DS_and_fine:
                            lossall = train_config.weight_D_S * loss + \
                                      model.ECE_weight_D_D * loss_D_D + \
                                      (1 - model.ECE_weight_D_D) * loss_S_S + \
                                      train_config.weight_D_fine_S_fine * loss_D_fine_S_fine + \
                                      model.ECE_weight_D_fine_D_fine * loss_D_fine_D_fine + \
                                      (1 - model.ECE_weight_D_fine_D_fine) * loss_S_fine_S_fine

                    elif train_config.if_use_multiply_1:
                        if train_config.only_DS:
                            lossall = train_config.weight_D_S * loss + \
                                      0.5 * model.ECE_weight_D_D * loss_D_D + \
                                      0.5 * (1/model.ECE_weight_D_D) * loss_S_S + \
                                      train_config.weight_D_fine_S_fine * loss_D_fine_S_fine + \
                                      0. * loss_D_fine_D_fine + \
                                      0. * loss_S_fine_S_fine

                        elif train_config.only_fine:
                            lossall = train_config.weight_D_S * loss + \
                                      0. * loss_D_D + \
                                      0. * loss_S_S + \
                                      train_config.weight_D_fine_S_fine * loss_D_fine_S_fine + \
                                      0.5 * model.ECE_weight_D_fine_D_fine * loss_D_fine_D_fine + \
                                      0.5 * (1/model.ECE_weight_D_fine_D_fine) * loss_S_fine_S_fine

                        elif train_config.DS_and_fine:
                            lossall = train_config.weight_D_S * loss + \
                                      0.5 * model.ECE_weight_D_D * loss_D_D + \
                                      0.5 * (1/model.ECE_weight_D_D) * loss_S_S + \
                                      train_config.weight_D_fine_S_fine * loss_D_fine_S_fine + \
                                      0.5 * model.ECE_weight_D_fine_D_fine * loss_D_fine_D_fine + \
                                      0.5 * (1/model.ECE_weight_D_fine_D_fine) * loss_S_fine_S_fine

                else:
                    lossall = train_config.weight_D_S * loss + \
                            train_config.weight_S_S * loss_S_S + \
                            train_config.weight_D_D * loss_D_D + \
                            train_config.weight_D_fine_S_fine * loss_D_fine_S_fine + \
                            train_config.weight_D_fine_D_fine * loss_D_fine_D_fine + \
                            train_config.weight_S_fine_S_fine * loss_S_fine_S_fine


                losses.update(lossall.item())

            # scaler.scale(loss).backward()
            scaler.scale(lossall).backward()
            # print(f"\n=================pos_scale:{model.model_1.pos_scale}====================")

            # Gradient clipping
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        else:

            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                loss = loss_functions["infoNCE"](features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_functions["infoNCE"](features1, features2, model.logit_scale.exp())
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()

            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        if train_config.verbose:
            # tst = model.logit_scale
            monitor = {
                "loss": "{:.4f}".format(loss.item()),
                "loss_avg": "{:.4f}".format(losses.avg),
                "lr": "{:.6f}".format(optimizer.param_groups[0]['lr'])}

            bar.set_postfix(ordered_dict=monitor)

            if tensorboard is not None:
                steps = step + (epoch - 1) * train_steps_per
                tensorboard.add_scalar("Loss", lossall.item(), steps)
                tensorboard.add_scalar("Loss_Avg", losses.avg, steps)
                tensorboard.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], steps)
                tensorboard.add_scalar("Learning_Rate_Temp", optimizer.param_groups[-1]['lr'], steps)
                tensorboard.add_scalar("Temperature", model.logit_scale.detach().cpu().numpy(), steps)

        step += 1
        # break

    if train_config.verbose:
        bar.close()

    print("/n================================================")
    print("D_D:{}", model.ECE_weight_D_D)
    # print("S_S:{}", model.ECE_weight_S_S)
    print("D_fine_D_fine:{}", model.ECE_weight_D_fine_D_fine)
    # print("S_fine_S_fine:{}", model.ECE_weight_S_fine_S_fine)
    print("================================================")
    return losses.avg



def predict(train_config, model, dataloader):
    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    img_features_list = []
    ids_list = []
    paths_list = []

    count = 0

    with torch.no_grad():

        for img, ids, paths in bar:
            count+=1

            ids_list.append(ids)

            with autocast():
                img = img.to(train_config.device)

                if train_config.handcraft_model is not True:
                    img_feature = model(img)
                else:
                    img_feature = model(img)[-2]
                    # img_fine_grained = model(img)[0]
                    # print()

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)

            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
            paths_list.append(paths)

            # if count >= 5:
            #     break

        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        paths_list = [item for sublist in paths_list for item in sublist]

    if train_config.verbose:
        bar.close()

    return img_features, ids_list, paths_list
