#!/usr/bin/env bash
# -*- coding: utf-8 -*-


GPU=$1
workdir=/srv/scratch6/kew/mbart/longmbart
data=$workdir/dummy/de/head100
finetuned=$workdir/dummy/de/finetuned/w512/
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
    --test_source $data/test.review \
    --test_target $data/test.response \
    --max_output_len 512 \
    --max_input_len 1024 \
    --batch_size 1 \
    --num_workers 5 \
    --gpus 1 \
    --beam_size 6 \
    --progress_bar_refresh_rate 1 \
    --src_lang de_DE --tgt_lang de_DE
    
# --tags_included