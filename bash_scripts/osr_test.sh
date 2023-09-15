#!/bin/bash
PYTHON='/home/oem/anaconda3/envs/open_set_mengyao/bin/python'
export CUDA_VISIBLE_DEVICES=0
# Get unique log file
SAVE_DIR=./output/

# SPECIFY PARAMS
DATASET="paddy_rice"
LOSS="ARPLoss"
#model="vit_large_patch16"
model="resnet50"
AMS_LOSS="False"

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

# if model is vit_large_patch16 and src_dset is vit_in1k_supervised_large, the pretrain="vit_in1k_supervised_large"
${PYTHON} -m methods.tests.openset_test_fine \
--model=${model} \
--loss=${LOSS} \
--pretrain="" \
--dataset=${DATASET} \
--image_size=336 \
--max_epoch=99 \
--admloss=${AMS_LOSS} \
> ${SAVE_DIR}logfile_${EXP_NUM}.txt

