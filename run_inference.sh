#!/usr/bin/env bash
# -*- coding: utf-8 -*-

#
# NOTE: adjust paths below to system before running
#
# Example call:
# bash run_inference.sh 1
#

set -e

GPU=$1
scratch=/srv/scratch6/kew/mbart/hospo_respo/respo_final/

data=$scratch/data/ # regular test set (2020)
finetuned=$scratch/mbart_model_2021-06-03/ft/2021-06-01_DR/
model_checkpoint=$finetuned/'checkpointepoch=19_vloss=3.56469.ckpt'
outdir=$finetuned/inference/
outfile=$outdir/translations.json

SRC="review_tagged_DER"

if [[ -z $GPU ]]; then
  echo "No GPU specified!" && exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU

mkdir -p $outdir

set -x

python inference.py \
    --model_path $finetuned \
    --checkpoint $model_checkpoint \
    --tokenizer $finetuned \
    --test_source $data/test.$SRC \
    --infer_target_tags \
    --tags_included \
    --max_output_len 512 \
    --max_input_len 512 \
    --batch_size 4 \
    --num_workers 5 \
    --gpus 1 \
    --beam_size 6 \
    --do_sample --top_k 5 \
    --progress_bar_refresh_rate 1 \
    --num_return_sequences 6 \
    --translation $outfile --output_to_json;