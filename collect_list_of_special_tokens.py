#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TODO: prepare corpus file
# use mBART spm model
# /srv/scratch6/kew/mbart/mbart.cc25.v2/sentence.bpe.model
# to check which tokens are to be kept
# write keep-tokens to outputfile

"""

Example call:

    python collect_list_of_special_tokens.py $data/train.review $data/train.response $data/train.rating $data/train.domain $data/train.est_label --outfile /srv/scratch6/kew/mbart/dummy_de/vocab.txt
    
"""

import argparse
from collections import Counter
from typing import List
import sentencepiece as sp

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus_files', nargs='+', help='list of corpus files to read and encode to find relevant sentencepiece pieces.')
    ap.add_argument('-o', '--outfile', help='path to output file.')
    return ap.parse_args()

def collect_special_tokens(infiles: List[str]):
    
    relevant_tokens = Counter()

    for infile in infiles:
        print(f'reading pieces from file {infile} ...')
        with open(infile, 'r', encoding='utf8') as inf:
            for line in inf:
                line = line.strip()
                tokens = [token for token in line.split() if token.startswith('<') and token.endswith('>')]
                relevant_tokens.update(tokens)

    print(f'collected {len(relevant_tokens)} tokens')

    return relevant_tokens    

def write_vocab_file(tokens, outfile):

    with open(outfile, 'w', encoding='utf8') as outf:
        for token in sorted(tokens.keys()):
            print('special token:', token)
            outf.write(f'{token}\n')
    return


if __name__ == "__main__":
    args = set_args()

    special_tokens = collect_special_tokens(args.corpus_files)
    
    # write pieces to vocab list
    write_vocab_file(special_tokens, args.outfile)

