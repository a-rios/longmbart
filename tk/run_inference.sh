#!/usr/bin/env bash
# -*- coding: utf-8 -*-


GPU=$1
scratch=/srv/scratch6/kew/mbart/longmbart
data=$scratch/dummy/de/raw/
finetuned=$scratch/dummy/de/finetuned/w512/
outdir=$finetuned/inference/

if [[ -z $GPU ]]; then
  echo "No GPU specified!" && exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU

mkdir -p $outdir

python -m longformer.simplify \
    --model_path $finetuned \
    --checkpoint "checkpointepoch=02_rougeL=0.00000.ckpt" \
    --tokenizer $finetuned \
    --translation $outdir/chkpt_E02.txt \
    --test_source $data/test.review_tagged \
    --test_target $data/test.response_tagged \
    --max_output_len 512 \
    --max_input_len 1024 \
    --batch_size 1 \
    --num_workers 5 \
    --gpus 1 \
    --beam_size 6 \
    --progress_bar_refresh_rate 1 \
    --tags_included


# --src_lang de_DE --tgt_lang de_DE
    
# 