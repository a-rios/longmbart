#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
from pathlib import Path

from filter_foreign import filter_foreign_characters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter-files",
        type=argparse.FileType("r"),
        nargs="+",
        help="Files from which to create a frequency list",
        metavar="PATH",
    )
    parser.add_argument(
        "--complete-vocab",
        type=argparse.FileType("r"),
        help="File containing the complete vocabulary to filter",
        metavar="PATH",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for filtered vocabularies",
        metavar="PATH",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        help="File prefix for output files",
        metavar="STRING",
    )
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        help="Vocabulary sizes of the output",
        metavar="INT",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Create a raw vocabulary from the filter files",
    )
    parser.add_argument(
        "--filtered",
        action="store_true",
        help="Create a unicode-filtered vocabulary from the filter files",
    )
    args = parser.parse_args()
    return args


def get_freq_list(frequency_files):
    def by_value(item):
        return item[1]

    freq_dict = dict()
    freq_list = list()
    for file in frequency_files:
        lines = file.readlines()
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
        if len(filtered) >= n:
            break
        if c in unfiltered:
            filtered.append(c)

    return sorted(filtered)


def create_raw_vocab(freq_list, output_dir, output_prefix):
    outfilename = output_dir / '{}.raw'.format(output_prefix)
    with open(outfilename, 'w') as outfile:
        for piece in freq_list:
            outfile.write(piece + "\n")


def create_filtered_vocab(freq_list, output_dir, output_prefix):
    filtered = filter_foreign_characters(freq_list)
    outfilename = output_dir / '{}.filtered'.format(output_prefix)
    with open(outfilename, 'w') as outfile:
        for piece in filtered:
            outfile.write(piece + "\n")


def main(args: argparse.Namespace):
    freq_list = get_freq_list(args.filter_files)
    if args.raw and args.output_dir and args.output_prefix:
        create_raw_vocab(freq_list, args.output_dir, args.output_prefix)
    if args.filtered and args.output_dir and args.output_prefix:
        create_filtered_vocab(freq_list, args.output_dir, args.output_prefix)
    if args.vocab_sizes and args.complete_vocab and args.output_dir and args.output_prefix:
        complete = args.complete_vocab.readlines()
        unfiltered = filter_foreign_characters(complete, return_set=True)
        for n in args.vocab_sizes:
            filtered = filter_by_frequency(unfiltered, freq_list, n)
            outfilename = args.output_dir / '{}.{}k'.format(args.output_prefix, int(n/1000))
            with open(outfilename, 'w') as outfile:
                for token in filtered:
                    outfile.write(token + '\n')


if __name__ == '__main__':
    args = parse_args()
    main(args)
