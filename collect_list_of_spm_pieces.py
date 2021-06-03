#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TODO: prepare corpus file
# use mBART spm model
# /srv/scratch6/kew/mbart/mbart.cc25.v2/sentence.bpe.model
# to check which tokens are to be kept
# write keep-tokens to outputfile

"""

Example call:

    python collect_list_of_spm_pieces.py /srv/scratch6/kew/mbart/dummy_de/raw/train.review /srv/scratch6/kew/mbart/dummy_de/raw/train.response --spm ../hf4mbart/sentencepiece.bpe.model --outfile /srv/scratch6/kew/mbart/dummy_de/vocab.txt

"""

import argparse
from collections import Counter
from typing import List
import sentencepiece as sp

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus_files', nargs='+', help='list of corpus files to read and encode to find relevant sentencepiece pieces.')
    ap.add_argument('--spm', help='path to original mBART sentencepiece model')
    ap.add_argument('-o', '--outfile', help='path to output file.')
    return ap.parse_args()

def collect_pieces(infiles: List[str], spm: sp.SentencePieceProcessor):
    
    relevant_pieces = Counter()

    for infile in infiles:
        print(f'reading pieces from file {infile} ...')
        with open(infile, 'r', encoding='utf8') as inf:
            for line in inf:
                line = line.strip()
                pieces = spm.encode_as_pieces(line)
                relevant_pieces.update(pieces)

    print(f'collected {len(relevant_pieces)} pieces')

    return relevant_pieces    

def write_vocab_file(pieces, outfile):

    with open(outfile, 'w', encoding='utf8') as outf:
        for piece, _ in pieces.most_common():
            outf.write(f'{piece}\n')
    return


if __name__ == "__main__":
    args = set_args()

    # load spm:
    spm = sp.SentencePieceProcessor(model_file=args.spm)
    print(f'loaded sentencepiece model from {args.spm}')
    # collect overlapping sentencepiece tokens from corpus/spm
    relevant_pieces = collect_pieces(args.corpus_files, spm)

    relevant_pieces.update(["ar_AR","cs_CZ","de_DE","en_XX","es_XX","et_EE","fi_FI","fr_XX","gu_IN","hi_IN","it_IT","ja_XX","kk_KZ","ko_KR","lt_LT","lv_LV","my_MM","ne_NP","nl_XX","ro_RO","ru_RU","si_LK","tr_TR","vi_VN","zh_CN"])

    # write pieces to vocab list
    write_vocab_file(relevant_pieces, args.outfile)

