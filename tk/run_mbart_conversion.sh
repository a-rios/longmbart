#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -e

home=/home/user/kew/
workdir=$home/INSTALLS/longmbart/
scratch=/srv/scratch6/kew/mbart/
data=/srv/scratch6/kew/respo_hospo_data/rrgen_210426/ml_hosp_re/raw #$scratch/longmbart/dummy/
spm_pieces=$data/ml_vocab_210426.txt

# collect list-of-spm-pieces
python collect_list_of_spm_pieces.py \
    $data/train.review_tagged $data/train.response_tagged \
    $data/train.rating $data/train.domain $data/train.est_label \
    $data/valid.review_tagged $data/valid.response_tagged \
    $data/valid.rating $data/valid.domain $data/valid.est_label \
    --spm $scratch/hf4mbart/sentencepiece.bpe.model \
    --outfile $spm_pieces

echo "saved spm pieces to $spm_pieces"

echo "converting mBART to LONGmBART..."

MAX_POS=1024
ATT_WIN=128

python $workdir/scripts/convert_mbart_to_longformerencoderdecoder.py \
    --base_model facebook/mbart-large-cc25 \
    --save_model_to $scratch/longmbart/longmbart_$MAX_POS/ \
    --attention_window $ATT_WIN \
    --reduce-to-vocab $spm_pieces \
    --cache_dir $scratch/hf4mbart \
    --max_pos $MAX_POS #\

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

# echo "saved spm pieces to $spm_pieces"

# echo "converting mBART to LONGmBART..."

# python $workdir/scripts/convert_mbart_to_longformerencoderdecoder.py \
#     --base_model facebook/mbart-large-cc25 \
#     --save_model_to $scratch/longmbart/longmbart2 \
#     --attention_window 512 \
#     --reduce-to-vocab $spm_pieces \
#     --cache_dir $scratch/hf4mbart \
#     --max_pos 1024 #\
    # --initialize_tags $langs