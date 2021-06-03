#!/usr/bin/env bash
# -*- coding: utf-8 -*-

#
# NOTE: adjust paths below to system before running
#
# Example call:
# bash run_mbart_conversion.sh
#

set -e

scratch=/srv/scratch6/kew/mbart/
data=$scratch/hospo_respo/respo_final/data

spm_pieces="$data/spm_pieces.txt"
spec_tokens="$data/special_tokens.txt"

# collect list-of-spm-pieces
python collect_list_of_spm_pieces.py \
    $data/train.review $data/train.response \
    $data/valid.review $data/valid.response \
    --spm $scratch/hf4mbart/sentencepiece.bpe.model \
    --outfile $spm_pieces

echo "saved spm pieces to $spm_pieces"

python collect_list_of_special_tokens.py \
    $data/train.review $data/train.response \
    $data/train.rating $data/train.domain $data/train.est_label \
    $data/valid.rating $data/valid.domain $data/valid.est_label \
    --outfile $spec_tokens

echo "saved special tokens $spec_tokens"

timestamp() {
  date +"%Y-%m-%d" # current time
}

save_pref="mbart_model_$(timestamp)"
outdir=$data/../$save_pref/
echo "output path for trimmed mBART model: $outdir"

echo "trimming mBART's embedding matrix..."

python trim_mbart.py \
    --base_model facebook/mbart-large-cc25 \
    --save_model_to $outdir \
    --reduce-to-vocab $spm_pieces \
    --cache_dir $scratch/hf4mbart \
    --add_special_tokens $spec_tokens

echo "done!"