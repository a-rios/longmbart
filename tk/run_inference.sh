#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -e

GPU=$1
scratch=/srv/scratch6/kew/mbart/hospo_respo/ml_hosp_re_unmasked_untok/
data=$scratch/raw/ # regular test set (2020)
# data=$scratch/raw2021/ # updated 2021 test set
outdir=$scratch/

# finetuned=$scratch/2021-04-30_12-40-05_w128-2021
# model_checkpoint=$finetuned/'checkpointepoch=19_vloss=3.54154.ckpt'

# finetuned=$scratch/2021-04-29_18-44-42_w128
# model_checkpoint=$finetuned/'checkpointepoch=19_vloss=3.57276.ckpt'

finetuned=$scratch/2021-05-02_22-27-31_w128-der_pref
model_checkpoint=$finetuned/'checkpointepoch=19_vloss=3.56469.ckpt'

outdir=$finetuned/inference/

if [[ -z $GPU ]]; then
  echo "No GPU specified!" && exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU

mkdir -p $outdir

set -x

python -m longformer.simplify \
    --model_path $finetuned \
    --checkpoint $model_checkpoint \
    --tokenizer $finetuned \
    --test_source $data/test.review_dom_est_rat \
    --test_target $data/test.response \
    --tags_included \
    --max_output_len 512 \
    --max_input_len 512 \
    --batch_size 3 \
    --num_workers 5 \
    --gpus 1 \
    --beam_size 6 \
    --progress_bar_refresh_rate 1 \
    --num_return_sequences 1 \
    --translation $outdir/chkpt_E19_decode_on_raw2020.jsonl --output_to_json \

# --infer_target_tags \

# --test_target $data/test.response \
#--translation $outdir/chkpt_E08_.jsonl 
# --src_lang de_DE --tgt_lang de_DE
    
# 


# checkpointepoch=08_rougeL=0.29971.ckpt