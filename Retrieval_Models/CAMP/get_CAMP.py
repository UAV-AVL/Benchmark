import os
import torch
import argparse
from dataclasses import dataclass
from Retrieval_Models.CAMP.sample4geo.model import TimmModel


@dataclass
class Configuration:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Train and Test on University-1652')

        parser.add_argument('--ckpt_path',
                            default=r'Retrieval_Models/CAMP/weights/weights_0.9446_for_U1652.pth',
                            type=str, help='path to pretrained checkpoint file')

        # Added for your modification
        parser.add_argument('--model', default='convnext_base.fb_in22k_ft_in1k_384', type=str, help='backbone model')
        # parser.add_argument('--model', default='convnext_base_22k_1k_384', type=str, help='backbone model')
        parser.add_argument('--handcraft_model', default=True, type=bool, help='use modified backbone')
        parser.add_argument('--img_size', default=384, type=int, help='input image size')
        parser.add_argument('--views', default=2, type=int, help='only supports 2 branches retrieval')
        parser.add_argument('--record', default=True, type=bool, help='use tensorboard to record training procedure')

        # Model Config
        parser.add_argument('--nclasses', default=701, type=int, help='U-1652场景的类别数')
        parser.add_argument('--block', default=2, type=int)
        parser.add_argument('--triplet_loss', default=0.3, type=float)
        parser.add_argument('--resnet', default=False, type=bool)

        # Our tricks
        parser.add_argument('--weight_infonce', default=1.0, type=float)
        parser.add_argument('--weight_triplet', default=0., type=float)
        parser.add_argument('--weight_cls', default=0., type=float)
        parser.add_argument('--weight_fine', default=0., type=float)
        parser.add_argument('--weight_channels', default=0., type=float)
        parser.add_argument('--weight_dsa', default=0., type=float)
        parser.add_argument('--pos_scale', default=0.6, type=float)
        parser.add_argument('--infoNCE_logit', default=3.65, type=float)
        # -- the weights of loss are learnable
        parser.add_argument('--weight_D_S', default=1.0, type=float)
        parser.add_argument('--weight_D_D', default=0., type=float)
        parser.add_argument('--weight_S_S', default=0., type=float)
        parser.add_argument('--weight_D_fine_S_fine', default=0., type=float)
        parser.add_argument('--weight_D_fine_D_fine', default=0., type=float)
        parser.add_argument('--weight_S_fine_S_fine', default=0., type=float)

        # =========================================================================
        parser.add_argument('--blocks_for_PPB', default=3, type=int)

        parser.add_argument('--if_learn_ECE_weights', default=True, type=bool)
        parser.add_argument('--learn_weight_D_D', default=0., type=float)
        parser.add_argument('--learn_weight_S_S', default=0., type=float)
        parser.add_argument('--learn_weight_D_fine_S_fine', default=1.0, type=float)
        parser.add_argument('--learn_weight_D_fine_D_fine', default=0.5, type=float)
        parser.add_argument('--learn_weight_S_fine_S_fine', default=0., type=float)

        parser.add_argument('--if_use_plus_1', default=False, type=bool)
        parser.add_argument('--if_use_multiply_1', default=True, type=bool)
        parser.add_argument('--only_DS', default=False, type=bool)
        parser.add_argument('--only_fine', default=True, type=bool)
        parser.add_argument('--DS_and_fine', default=False, type=bool)

        # --
        parser.add_argument('--only_test', default=True, type=bool, help='use pretrained model to test')
        parser.add_argument('--only_draw_heat', default=False, type=bool, help='use pretrained model to test')


        # Training Config
        parser.add_argument('--mixed_precision', default=True, type=bool)
        parser.add_argument('--custom_sampling', default=True, type=bool)
        parser.add_argument('--seed', default=1, type=int, help='random seed')
        parser.add_argument('--epochs', default=1, type=int, help='1 epoch for 1652')
        parser.add_argument('--batch_size', default=24, type=int, help='remember the bs is for 2 branches')
        parser.add_argument('--verbose', default=True, type=bool)
        parser.add_argument('--gpu_ids', default=(0, 1, 2, 3), type=tuple)

        # Eval Config
        parser.add_argument('--batch_size_eval', default=128, type=int)
        parser.add_argument('--eval_every_n_epoch', default=1, type=int)
        parser.add_argument('--normalize_features', default=True, type=bool)
        parser.add_argument('--eval_gallery_n', default=-1, type=int)

        # Optimizer Config
        parser.add_argument('--clip_grad', default=100.0, type=float)
        parser.add_argument('--decay_exclue_bias', default=False, type=bool)
        parser.add_argument('--grad_checkpointing', default=False, type=bool)

        # Loss Config
        parser.add_argument('--label_smoothing', default=0.1, type=float)

        # Learning Rate Config
        parser.add_argument('--lr', default=0.001, type=float, help='1 * 10^-4 for ViT | 1 * 10^-1 for CNN')
        parser.add_argument('--scheduler', default="cosine", type=str, help=r'"polynomial" | "cosine" | "constant" | None')
        parser.add_argument('--warmup_epochs', default=0.1, type=float)
        parser.add_argument('--lr_end', default=0.0001, type=float)

        # Learning part Config
        parser.add_argument('--lr_mlp', default=None, type=float)
        parser.add_argument('--lr_decouple', default=None, type=float)
        parser.add_argument('--lr_blockweights', default=2, type=float)
        parser.add_argument('--lr_weight_ECE', default=None, type=float)

        # Dataset Config
        parser.add_argument('--dataset', default='U1652-D2S', type=str, help="'U1652-D2S' | 'U1652-S2D'")
        parser.add_argument('--data_folder', default='./data/U1652', type=str)
        parser.add_argument('--dataset_name', default='U1652', type=str)

        # Augment Images Config
        parser.add_argument('--prob_flip', default=0.5, type=float, help='flipping the sat image and drone image simultaneously')

        # Savepath for model checkpoints Config
        parser.add_argument('--model_path', default='./checkpoints/university', type=str)

        # Eval before training Config
        parser.add_argument('--zero_shot', default=False, type=bool)

        # Checkpoint to start from Config
        parser.add_argument('--checkpoint_start', default=None)

        # Set num_workers to 0 if on Windows Config
        parser.add_argument('--num_workers', default=0 if os.name == 'nt' else 4, type=int)

        # Train on GPU if available Config
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)

        # For better performance Config
        parser.add_argument('--cudnn_benchmark', default=True, type=bool)

        # Make cudnn deterministic Config
        parser.add_argument('--cudnn_deterministic', default=False, type=bool)

        args = parser.parse_args(namespace=self)


# -----------------------------------------------------------------------------#
# Train Config                                                                #
# -----------------------------------------------------------------------------#
def get_CAMP_model():
    config = Configuration()

    if config.handcraft_model is not True:
        # print("\nModel: {}".format(config.model))
        model = TimmModel(config.model,
                          pretrained=True, )

    else:
        from Retrieval_Models.CAMP.sample4geo.hand_convnext.model import make_model

        model = make_model(config)



    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)

    # Load pretrained Checkpoint
    if config.checkpoint_start is not None:
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=True)

        # Data parallel
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # Model to device
    model = model.to(config.device)

    best_score = 0

    checkpoint = torch.load(config.ckpt_path)

    if 1:
        del checkpoint['model_1.classifier1.classifier.0.weight']
        del checkpoint['model_1.classifier1.classifier.0.bias']
        del checkpoint['model_1.classifier_mcb1.classifier.0.weight']
        del checkpoint['model_1.classifier_mcb1.classifier.0.bias']
        del checkpoint['model_1.classifier_mcb2.classifier.0.weight']
        del checkpoint['model_1.classifier_mcb2.classifier.0.bias']
    model.load_state_dict(checkpoint, strict=False)

    return model



