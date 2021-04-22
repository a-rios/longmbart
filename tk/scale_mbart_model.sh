#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -e

home=/home/user/kew/
workdir=$home/INSTALLS/longmbart/
scratch=/srv/scratch6/kew/mbart/
data=$scratch/longmbart/dummy/
spm_pieces=$data/ml_vocab.txt

# langs=de_DE,en_XX

# # collect list-of-spm-pieces
# python collect_list_of_spm_pieces.py \
#     $data/de/raw/train.review $data/de/raw/train.response \
#     $data/de/raw/valid.review $data/de/raw/valid.response \
#     $data/de/raw/train.rating $data/de/raw/train.domain \
#     $data/de/raw/train.review $data/de/raw/train.source \
#     $data/en/raw/train.review $data/en/raw/train.response \
#     $data/en/raw/valid.review $data/en/raw/valid.response \
#     $data/en/raw/train.rating $data/en/raw/train.domain \
#     $data/en/raw/train.review $data/en/raw/train.source \
#     --spm $scratch/hf4mbart/sentencepiece.bpe.model \
#     --outfile $spm_pieces

python $workdir/scripts/convert_mbart_to_longformerencoderdecoder.py \
    --base_model facebook/mbart-large-cc25 \
    --save_model_to $scratch/scaled_mbart \
    --attention_window 512 \
    --reduce-to-vocab $spm_pieces \
    --cache_dir $scratch/hf4mbart \
    --max_pos 1024 #\
    # --initialize_tags $langs