# ----------------------
# PROJECT ROOT DIR
# ----------------------
project_root_dir = './'

# ----------------------
# EXPERIMENT SAVE PATHS
# ----------------------
exp_root = './output'  # directory to store experiment output (checkpoints, logs, etc)
save_dir = './output/baseline/ensemble_entropy_test'  # Evaluation save dir

# evaluation model path
root_model_path = './output/methods/ARPL/log/{}/arpl_models/{}/checkpoints/{}_{}_{}.pth'
root_criterion_path = './output/methods/ARPL/log/{}/arpl_models/{}/checkpoints/{}_{}_{}_criterion.pth'

# -----------------------
# DATASET ROOT DIRS
# -----------------------

cottonweed_train_root = './datasets/CottonWeedID15/raw/train'
cottonweed_val_root = './datasets/CottonWeedID15/raw/val'

ivadltomato_train_root = './datasets/IVADL_tomato/raw/train'
ivadltomato_val_root = './datasets/IVADL_tomato/raw/val'

ivadlrose_train_root = './datasets/IVADL_rose/raw/train'
ivadlrose_val_root = './datasets/IVADL_rose/raw/val'

paddyrice_train_root = './datasets/PaddyDoctor10407/raw/train'
paddyrice_val_root = './datasets/PaddyDoctor10407/raw/val'

# ----------------------
# PRETRAINED RESNET50 MODEL PATHS (For FGVC experiments)
# Weights can be downloaded from https://github.com/nanxuanzhao/Good_transfer
# ----------------------
imagenet_moco_path = './pretrained_models/moco_v2_imagenet.pth'
places_moco_path = './pretrained_models/moco_v2_places.pth'
coco_moco_path = './pretrained_models/moco_v2_coco.pth'
places_supervised_path = './pretrained_models/supervised_places.pth'
imagenet_supervised_path = './pretrained_models/supervised_imagenet.pth'
imagenet_mae_base_path = './pretrained_models/mae_pretrain_vit_base.pth'
imagenet_mae_large_path = './pretrained_models/mae_pretrain_vit_large.pth'
vit_in1k_supervised_large_path = './pretrained_models/imagenet21k+imagenet2012_ViT-L_16-224.npz'
vit_in21k_supervised_large_path = './pretrained_models/L_16-i21k.npz'
vit_in1k_mae_plantclef_softmax_large_path = './pretrained_models/ViT_IN1k_MAE_PlantCLEF2022_softmax_epoch100.pth'

