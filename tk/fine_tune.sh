#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -e

GPU=$1
workdir=/srv/scratch6/kew/mbart/longmbart
pretrained=$workdir/longmbart
data=$workdir/dummy/de/raw
save_dir=$workdir/dummy/de/finetuned/

MAX_TGT_LEN=1024
MAX_SRC_LEN=1024

if [[ -z $GPU ]]; then
  echo "No GPU specified!" && exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU

python -m longformer.simplification \
--from_pretrained $pretrained \
--tokenizer $pretrained \
--save_dir $save_dir \
--save_prefix "w512" \
--train_source $data/train.review \
--train_target $data/train.response \
--val_source $data/valid.review \
--val_target $data/valid.response \
--test_source $data/test.review \
--test_target $data/test.response \
--src_lang de_DE --tgt_lang de_DE \
--max_input_len $MAX_SRC_LEN --max_output_len $MAX_TGT_LEN \
--batch_size 1 \
--grad_accum 60 \
--num_workers 5 \
--gpus 1 \
--seed 222 \
--attention_dropout 0.1 \
--dropout 0.3 \
--attention_mode sliding_chunks \
--attention_window 512 \
--label_smoothing 0.2 \
--lr 0.00003 \
--val_every 1.0 \
--val_percent_check 1.0 \
--test_percent_check 1.0 \
--early_stopping_metric 'rougeL' \
--patience 10 \
--lr_reduce_patience 8 \
--lr_reduce_factor 0.5 \
--grad_ckpt \
--progress_bar_refresh_rate 10