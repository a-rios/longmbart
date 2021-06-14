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

data=$scratch/data/ # regular test set
finetuned=$scratch/mbart_model_2021-06-04/ft/2021-06-04_15-27-39
model_checkpoint=$finetuned/'checkpointepoch=00_vloss=4.26919.ckpt'
outdir=$finetuned/inference/
outfile=$outdir/translations.json

SRC="review_tagged"
TGT="response_tagged"

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
    --test_target $data/test.$TGT \
    --tags_included \
    --max_input_len 512 \
    --max_output_len 400 \
    --batch_size 4 \
    --num_workers 5 \
    --gpus 1 \
    --beam_size 5 \
    --progress_bar_refresh_rate 1 \
    --num_return_sequences 1 \
    --translation $outfile --output_to_json;