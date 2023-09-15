#!/bin/bash
PYTHON="/home/oem/anaconda3/envs/open_set_mengyao/bin/python"
export CUDA_VISIBLE_DEVICES=2
# Get unique log file
SAVE_DIR=./output/
AMS_LOSS='False'
AUG_M=30
AUG_N=2
LABEL_SMOOTHING=0
LR=0.0001

#############################################################
# VIT_PlantCLEF
#############################################################
#export model='vit_large_patch16'  #
#export src_dset="vit_in1k_mae_plantclef_softmax_large"  #   vit_in1k_supervised_large imagenet_large
#export tgt_dset="ivadl_tomato"
#export batch_size=32
#export optim="sgd"
#export epoch=100
##export SPLIT_IDX=2
#for SPLIT_IDX in 2 ; do
#  EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
#  EXP_NUM=$((${EXP_NUM}+1))
#  echo $EXP_NUM
#
#  ${PYTHON} -m methods.ARPL.osr \
#  --transform='rand-augment' --rand_aug_m=${AUG_M} --rand_aug_n=${AUG_N} \
#  --label_smoothing=${LABEL_SMOOTHING} \
#  --split_idx=${SPLIT_IDX} \
#  --batch_size=${batch_size} \
#  --model=${model} --feat_dim=1000 --image_size=224 \
#  --pretrain=${src_dset} --dataset=${tgt_dset} \
#  --max-epoch=${epoch} \
#  --admloss=${AMS_LOSS} \
#  --lr=${LR} \
#  --expr_name "${model}_src_${src_dset}_tgt_${tgt_dset}_split${SPLIT_IDX}_epoch${epoch}_adm${AMS_LOSS}_${LR}_revise" \
#  > ${SAVE_DIR}logfile_${EXP_NUM}.txt
#done

#############################################################
# VIT_ImageNet
#############################################################
#export model='vit_large_patch16'
#export src_dset="vit_in1k_mae_plantclef_softmax_large"  # imagenet_large
#export tgt_dset="paddy_rice"
#export batch_size=32
#export optim="sgd"
#export epoch=100
##export SPLIT_IDX=2
#
#for SPLIT_IDX in 0 1 2 3 4 ; do
#  EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
#  EXP_NUM=$((${EXP_NUM}+1))
#  echo $EXP_NUM
#
#  ${PYTHON} -m methods.ARPL.osr \
#  --transform='rand-augment' --rand_aug_m=${AUG_M} --rand_aug_n=${AUG_N} \
#  --label_smoothing=${LABEL_SMOOTHING} \
#  --split_idx=${SPLIT_IDX} \
#  --batch_size=${batch_size} \
#  --model=${model} --feat_dim=1000 --image_size=224 \
#  --pretrain=${src_dset} --dataset=${tgt_dset} \
#  --max-epoch=${epoch} \
#  --admloss=${AMS_LOSS} \
#  --lr=${LR} \
#  --expr_name "${model}_src_${src_dset}_tgt_${tgt_dset}_split${SPLIT_IDX}_epoch${epoch}_ams${AMS_LOSS}${LR}_all_openness1" \
#  > ${SAVE_DIR}logfile_${EXP_NUM}.txt
#done


#############################################################
# CNN_ResNet50
###########################################################
export model='resnet50'
export src_dset="imagenet"  #  imagenet_moco
export tgt_dset="paddy_rice"
export batch_size=32
export optim="sgd"
export epoch=100

for SPLIT_IDX in 0 1 2 3 4 ; do
  EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
  EXP_NUM=$((${EXP_NUM}+1))
  echo $EXP_NUM

  ${PYTHON} -m methods.ARPL.osr \
  --transform='rand-augment' --rand_aug_m=${AUG_M} --rand_aug_n=${AUG_N} \
  --label_smoothing=${LABEL_SMOOTHING} \
  --split_idx=${SPLIT_IDX} \
  --batch_size=${batch_size} \
  --model=${model} --feat_dim=2048 --image_size=336 \
  --pretrain=${src_dset} --dataset=${tgt_dset} \
  --max-epoch=${epoch} \
  --admloss=${AMS_LOSS} \
  --lr=${LR} \
  --expr_name "${model}_src_${src_dset}_tgt_${tgt_dset}_split${SPLIT_IDX}_epoch${epoch}_ams${AMS_LOSS}_${LR}_revise336" \
  > ${SAVE_DIR}logfile_${EXP_NUM}.txt
done


#############################################################
# ResNet50
#############################################################
#export model='resnet50'
#export src_dset="imagenet"
#export tgt_dset="paddy_rice"
#export batch_size=32
#export optim="sgd"
#export epoch=100
##export SPLIT_IDX=2
#
#for SPLIT_IDX in 0 1 2 3 4; do
#  EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
#  EXP_NUM=$((${EXP_NUM}+1))
#  echo $EXP_NUM
#
#  ${PYTHON} -m methods.ARPL.osr \
#  --transform='rand-augment' --rand_aug_m=${AUG_M} --rand_aug_n=${AUG_N} \
#  --label_smoothing=${LABEL_SMOOTHING} \
#  --split_idx=${SPLIT_IDX} \
#  --batch_size=${batch_size} \
#  --model=${model} --feat_dim=2048 --image_size=224 \
#  --pretrain=${src_dset} \
#  --dataset=${tgt_dset} \
#  --max-epoch=${epoch} \
#  --admloss=${AMS_LOSS} \
#  --lr=${LR} \
#  --expr_name "${model}_src_${src_dset}_tgt_${tgt_dset}_split${SPLIT_IDX}_epoch${epoch}_adm${AMS_LOSS}_${LR}_all" \
#  > ${SAVE_DIR}logfile_${EXP_NUM}.txt
#done