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
    --checkpoint "checkpointepoch=08_rougeL=0.29971.ckpt" \
    --tokenizer $finetuned \
    --test_source $data/test.review_tagged \
    --test_target $data/test.response_tagged \
    --max_output_len 512 \
    --max_input_len 512 \
    --batch_size 2 \
    --num_workers 5 \
    --gpus 1 \
    --beam_size 2 \
    --progress_bar_refresh_rate 1 \
    --tags_included \
    --temperature 1.2 --num_return_sequences 2 \
    --output_to_json

#--translation $outdir/chkpt_E08_.jsonl 
# --src_lang de_DE --tgt_lang de_DE
    
# 