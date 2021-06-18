#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import os            
            
from filter_foreign import filter_foreign_characters


def get_freq_list(filter_data_dir):
    def by_value(item):
        return item[1]
        
    freq_dict = dict()
    freq_list = list()
    # frequency_files = os.listdir(filter_data_dir)
    frequency_files = ['APA_capito_webcorpus.spm']
    for file in frequency_files:
        with open(os.path.join(filter_data_dir, file), 'r') as infile:
            lines = infile.readlines()
        for line in lines:
            tokens = line.split()
            for token in tokens:
                if token in freq_dict:
                    freq_dict[token] += 1
                else:
                    freq_dict[token] = 1
    
    for k, v in sorted(freq_dict.items(), key=by_value, reverse=True):
        freq_list.append(k)
    return freq_list
    
    
def filter_by_frequency(unfiltered, freq_list, n):
    filtered = list()
    
    for c in freq_list:
        if len(filtered) >= n: break
        if c in unfiltered:
            filtered.append(c)
    
    return sorted(filtered)
    
    
def main(filter_data_dir, vocab_dir, filename, n_list):
    freq_list = get_freq_list(filter_data_dir)
    with open(os.path.join(vocab_dir, filename), 'r') as infile:
        complete = infile.readlines()
    unfiltered = filter_foreign_characters(complete, return_set=True)
    for n in n_list:
        filtered = filter_by_frequency(unfiltered, freq_list, n)
        outfilename = os.path.join(vocab_dir, '{}.{}k'.format(filename, int(n/1000)))
        with open(outfilename, 'w') as outfile:
            for token in filtered:
                outfile.write(token + '\n')

                
if __name__ == '__main__':
    filter_data_dir = '../data/frequency_data'
    vocab_dir = '../data/vocabulary'
    filename = 'all.spm.uniq'
    n_list = [20000, 23000, 25000, 27000, 30000, 35000]
    main(filter_data_dir, vocab_dir, filename, n_list)