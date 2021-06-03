#!/usr/bin/env bash
# -*- coding: utf-8 -*-

#
# NOTE: adjust paths below to system before running
#
# Example call:
# nohup bash run_finetuning 4 > logs/finetuning.log &
#

set -e

GPU=$1
scratch=/srv/scratch6/kew/mbart/hospo_respo/respo_final/
pretrained=$scratch/mbart_model_2021-06-03/
data=$scratch/data/
outdir=$pretrained/ft/

## params
SRC=review_tagged
TGT=response_tagged
MAX_SRC_LEN=512
MAX_TGT_LEN=512

if [[ -z $GPU ]]; then
  echo "No GPU specified!" && exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

save_pref="$(timestamp)"
echo "Fine-tuning output dir: $outdir/$save_pref"

echo "Running on GPU(s) $GPU"

set -x # to log experiment execution

python train.py \
--from_pretrained $pretrained \
--tokenizer $pretrained \
--save_dir $outdir \
--save_prefix 2021-06-03_DER \
--train_source $data/train.$SRC --train_target $data/train.$TGT \
--val_source $data/valid.$SRC --val_target $data/valid.$TGT \
--test_source $data/test.$SRC --test_target $data/test.$TGT \
--tags_included \
--max_input_len $MAX_SRC_LEN --max_output_len $MAX_TGT_LEN \
--batch_size 8 \
--grad_accum 5 \
--num_workers 5 \
--gpus 1 \
--seed 222 \
--attention_dropout 0.1 \
--dropout 0.3 \
--label_smoothing 0.2 \
--lr 0.00003 \
--val_every 1.0 \
--val_percent_check 1.0 \
--test_percent_check 1.0 \
--early_stopping_metric vloss \
--patience 5 --max_epochs 20 \
--lr_reduce_patience 8 --lr_reduce_factor 0.5 \
--grad_ckpt \
--progress_bar_refresh_rate 1 \
--save_top_k 5 \
--wandb readvisor;
