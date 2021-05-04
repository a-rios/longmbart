#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -e

data=/srv/scratch6/kew/mbart/hospo_respo/ml_hosp_re_unmasked_untok
N_train=50
N_val_test=$((N_train/10))

outdir=$data/head$N_train
mkdir -p $outdir

head -n $N_train $data/raw/train.review > $outdir/train.review
head -n $N_train $data/raw/train.response > $outdir/train.response

head -n $N_val_test $data/raw/valid.review > $outdir/valid.review
head -n $N_val_test $data/raw/valid.response > $outdir/valid.response

head -n $N_val_test $data/raw/test.review > $outdir/test.review
head -n $N_val_test $data/raw/test.response > $outdir/test.response

echo ""
echo "done! Output: $outdir"
echo ""
