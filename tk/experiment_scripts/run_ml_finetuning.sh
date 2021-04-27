#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -e

GPU=$1
scratch=/srv/scratch6/kew/mbart/longmbart
pretrained=$scratch/longmbart
data=/srv/scratch6/kew/respo_hospo_data/rrgen_210426/ml_hosp_re/raw
# data=$scratch/dummy/en/raw/
save_dir=$scratch/ml_mbart/finetuned

# FORMAT=''#_tagged
MAX_TGT_LEN=512
MAX_SRC_LEN=512
ATT_WIN=128

if [[ -z $GPU ]]; then
  echo "No GPU specified!" && exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

save_pref="$(timestamp)_w$ATT_WIN"
echo "Fine-tuning output dir: $save_dir/$save_pref"

echo "Running on GPU(s) $GPU"

python -m longformer.simplification \
--from_pretrained $pretrained \
--tokenizer $pretrained \
--save_dir $save_dir \
--save_prefix $save_pref \
--train_source $data/train.review_tagged_pref_dom_rat \
--train_target $data/train.response_tagged \
--val_source $data/valid.review_tagged_pref_dom_rat \
--val_target $data/valid.response_tagged \
--test_source $data/test.review_tagged_pref_dom_rat \
--test_target $data/test.response_tagged \
--tags_included \
--max_input_len $MAX_SRC_LEN --max_output_len $MAX_TGT_LEN \
--batch_size 4 \
--grad_accum 10 \
--num_workers 5 \
--gpus 1 \
--seed 222 \
--attention_dropout 0.1 \
--dropout 0.3 \
--attention_mode sliding_chunks \
--attention_window $ATT_WIN \
--label_smoothing 0.2 \
--lr 0.00003 \
--val_every 1.0 \
--val_percent_check 1.0 \
--test_percent_check 1.0 \
--early_stopping_metric vloss \
--patience 5 --max_epochs 20 \
--lr_reduce_patience 8 --lr_reduce_factor 0.5 \
--grad_ckpt \
--progress_bar_refresh_rate 1 

# --src_lang en_XX --tgt_lang en_XX \
# --src_lang de_DE --tgt_lang de_DE \
# --tags_included \